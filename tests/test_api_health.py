# tests/test_api_health.py
"""
Tests for api.py's ops endpoints and — more importantly — for the bug
found while adding them.

The bug: every route was declared with `@app.get(...)` against the
module-level `app = create_app()` object, while main.py's start_sync_api()
serves a *different* instance from its own create_app(api_key=...) call.
A decorator only attaches a route to the one object it names, so the app
actually being served had no /health, no /sync/pull and no /sync/push —
just an empty FastAPI shell and its /docs page. Nothing raised; callers
got 404s with no explanation, and the code read as though the routes
existed.

`test_create_app_instances_expose_every_route` is the regression test for
that, and it's the most valuable test in this file — it fails the moment
anyone adds a route with a decorator bound to `app` again.

Also covers /health/modules (backlog #19) and the registry version fields
(#11).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api import create_app  # noqa: E402
from modules.hecate.intent_registry import registry_info  # noqa: E402

_EXPECTED_PATHS = {"/health", "/health/modules", "/sync/pull", "/sync/push"}


class _FakeDiagnostics:
    def __init__(self, status=None, raises=False):
        self._status = status if status is not None else {}
        self._raises = raises

    def module_status(self):
        if self._raises:
            raise RuntimeError("registry unavailable")
        return self._status


class _FakeNLU:
    def __init__(self, raises=False):
        self._raises = raises

    def cache_stats(self):
        if self._raises:
            raise RuntimeError("no cache")
        return {"entries": 4, "hits": 9, "misses": 3, "hit_rate": 0.75}


def _paths(app) -> set[str]:
    """
    All route paths reachable on *app*.

    Not a simple walk of app.routes: newer FastAPI wraps an included
    router in an _IncludedRouter object that holds the real routes behind
    `original_router` rather than exposing them as a `routes` list. Older
    versions splice the routes in directly. Handle both, plus arbitrary
    nesting, so this test asserts on what's actually served rather than on
    one version's internal layout.
    """
    found: set[str] = set()
    seen: set[int] = set()

    def walk(routes):
        for route in routes or []:
            if id(route) in seen:
                continue
            seen.add(id(route))

            path = getattr(route, "path", None)
            if isinstance(path, str) and path:
                found.add(path)

            nested = getattr(route, "routes", None)
            if nested:
                walk(nested)

            inner = getattr(route, "original_router", None)
            if inner is not None:
                walk(getattr(inner, "routes", None))

    walk(app.routes)
    return found


# ---------------------------------------------------------------------------
# The route-registration bug
# ---------------------------------------------------------------------------

def test_create_app_instances_expose_every_route():
    # Regression test for the decorator-bound-to-the-wrong-app bug.
    assert _EXPECTED_PATHS <= _paths(create_app())


def test_a_second_independent_instance_also_has_the_routes():
    # main.py serves an instance built with an api_key, not the module
    # level one; both must be complete.
    assert _EXPECTED_PATHS <= _paths(create_app(api_key="secret"))


def test_module_level_app_has_the_routes_too():
    import api

    assert _EXPECTED_PATHS <= _paths(api.app)


def test_instances_are_distinct_objects():
    assert create_app() is not create_app()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_ok_and_the_device_name():
    client = TestClient(create_app(device_name="laptop"))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["device"] == "laptop"
    assert body["timestamp"]


def test_health_includes_the_registry_version_and_fingerprint():
    # Backlog #11: this is the cheap endpoint a client polls to notice the
    # intent set changed under it.
    body = TestClient(create_app()).get("/health").json()
    info = registry_info()
    assert body["registry_version"] == info["version"]
    assert body["registry_fingerprint"] == info["fingerprint"]


def test_health_needs_no_api_key_even_when_one_is_configured():
    # Liveness has to be pollable by a monitor that holds no secret; only
    # /sync/* is guarded.
    client = TestClient(create_app(api_key="secret"))
    assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# /health/modules (#19)
# ---------------------------------------------------------------------------

def test_modules_health_without_diagnostics_reports_unknown():
    body = TestClient(create_app()).get("/health/modules").json()
    assert body["status"] == "unknown"
    assert body["module_count"] == 0
    assert body["modules"] == {}


def test_modules_health_reports_ok_when_all_modules_are_ready():
    diag = _FakeDiagnostics({
        "athena": {"registered": True, "state": "ready", "probe": "ready"},
        "iris": {"registered": True, "state": "ready", "probe": "available"},
    })
    body = TestClient(create_app(diagnostics=diag)).get("/health/modules").json()
    assert body["status"] == "ok"
    assert body["module_count"] == 2
    assert body["modules"]["athena"]["probe"] == "ready"


def test_probeless_modules_do_not_drag_the_aggregate_down():
    # Most modules expose no health method; that's normal, not a fault.
    diag = _FakeDiagnostics({
        "core": {"registered": True, "state": "unknown", "probe": None},
        "athena": {"registered": True, "state": "ready", "probe": "ready"},
    })
    body = TestClient(create_app(diagnostics=diag)).get("/health/modules").json()
    assert body["status"] == "ok"


def test_one_degraded_module_degrades_the_aggregate():
    diag = _FakeDiagnostics({
        "athena": {"registered": True, "state": "ready", "probe": "ready"},
        "iris": {"registered": True, "state": "degraded", "probe": "available"},
    })
    body = TestClient(create_app(diagnostics=diag)).get("/health/modules").json()
    assert body["status"] == "degraded"


def test_an_errored_module_degrades_the_aggregate_and_keeps_its_detail():
    diag = _FakeDiagnostics({
        "pluto": {
            "registered": True, "state": "error",
            "probe": "available", "detail": "postgres pool exhausted",
        },
    })
    body = TestClient(create_app(diagnostics=diag)).get("/health/modules").json()
    assert body["status"] == "degraded"
    assert "postgres" in body["modules"]["pluto"]["detail"]


def test_modules_health_never_500s_when_diagnostics_raises():
    # A monitoring endpoint that throws is worse than one that says
    # "unknown".
    client = TestClient(create_app(diagnostics=_FakeDiagnostics(raises=True)))
    response = client.get("/health/modules")
    assert response.status_code == 200
    assert response.json()["status"] == "unknown"


def test_modules_health_includes_nlu_cache_stats_when_available():
    client = TestClient(create_app(diagnostics=_FakeDiagnostics(), nlu=_FakeNLU()))
    body = client.get("/health/modules").json()
    assert body["nlu_cache"]["hit_rate"] == 0.75


def test_modules_health_omits_cache_stats_when_the_nlu_raises():
    client = TestClient(
        create_app(diagnostics=_FakeDiagnostics(), nlu=_FakeNLU(raises=True))
    )
    body = client.get("/health/modules").json()
    assert body["nlu_cache"] is None


def test_modules_health_omits_cache_stats_when_no_nlu_is_supplied():
    body = TestClient(create_app(diagnostics=_FakeDiagnostics())).get(
        "/health/modules"
    ).json()
    assert body["nlu_cache"] is None


def test_modules_health_includes_registry_info():
    body = TestClient(create_app()).get("/health/modules").json()
    assert body["registry"]["intent_count"] == registry_info()["intent_count"]


# ---------------------------------------------------------------------------
# Auth still guards /sync/* only
# ---------------------------------------------------------------------------

def test_sync_requires_the_api_key_when_configured():
    client = TestClient(create_app(api_key="secret"))
    assert client.get("/sync/pull").status_code == 401


def test_sync_accepts_the_api_key_header():
    # No memory attached, so this gets as far as the memory dependency's
    # 503 — which proves auth passed and the route exists.
    client = TestClient(create_app(api_key="secret"))
    response = client.get("/sync/pull", headers={"X-API-Key": "secret"})
    assert response.status_code == 503


def test_sync_without_a_configured_key_is_unauthenticated():
    client = TestClient(create_app())
    assert client.get("/sync/pull").status_code == 503
