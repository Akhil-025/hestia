# api.py

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, FastAPI, Request, HTTPException, Depends, Header, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import time

from modules.hecate.intent_registry import registry_info

logger = logging.getLogger(__name__)

# Every route is declared on this router and included by create_app()
# below, NOT on a module-level `app` object.
#
# This was a real bug, found while adding /health/modules: the routes used
# to be declared with `@app.get(...)` decorators against the module-level
# `app = create_app()` instance, while main.py's start_sync_api() serves a
# *different* instance built by its own `create_app(api_key=...)` call.
# Decorators only attach a route to the one object they name, so the app
# actually being served had no /health, no /sync/pull and no /sync/push at
# all — just the empty FastAPI shell and its /docs page. Anything calling
# the sync API got a 404 and there was nothing in the logs to explain why,
# because the code plainly *looked* like it declared those routes.
#
# An APIRouter fixes it structurally: routes belong to the router, and
# every app built by create_app() gets them.
router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Interaction(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    response: str = Field(..., min_length=1, max_length=32768)
    intent: str = Field(..., min_length=1, max_length=256)
    pushed_at: Optional[str] = None

    @field_validator("query", "response", "intent", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return v


class PushRequest(BaseModel):
    interactions: list[Interaction] = Field(..., max_length=1000)


class PullResponse(BaseModel):
    interactions: list[dict]
    timestamp: str
    count: int


class PushResponse(BaseModel):
    status: str
    merged: int
    skipped: int


class HealthResponse(BaseModel):
    status: str
    device: str
    timestamp: str
    registry_version: Optional[str] = None
    registry_fingerprint: Optional[str] = None


class ModuleHealth(BaseModel):
    """Per-module health as normalised by Diagnostics.module_status()."""

    registered: bool
    state: str          # ready | degraded | unknown | error
    probe: Optional[str] = None
    detail: Optional[str] = None


class ModulesHealthResponse(BaseModel):
    """
    Aggregate health for the web UI to poll (backlog #19).

    `status` is the rolled-up verdict across every module:
      ok        every module reports ready, or exposes no probe
      degraded  at least one module is degraded or errored
      unknown   diagnostics aren't wired up (no orchestrator bound)
    """

    status: str
    device: str
    timestamp: str
    module_count: int
    modules: dict[str, ModuleHealth]
    registry: dict
    nlu_cache: Optional[dict] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up")
    yield
    logger.info("API shutting down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    device_name: str = "laptop",
    api_key: Optional[str] = None,
    cors_origins: Optional[list[str]] = None,
    diagnostics: Optional[object] = None,
    nlu: Optional[object] = None,
) -> FastAPI:
    """
    api_key: optional shared secret required (via X-API-Key header or
        ?api_key= query param) on every /sync/* request. If unset, /sync/*
        is unauthenticated — the caller (main.py's start_sync_api) is
        responsible for only allowing that when the server is bound to
        loopback, the same contract web_ui.py enforces for /api/*.
    diagnostics: optional core.observability.Diagnostics instance. When
        supplied, /health/modules reports every registered module's state;
        without it that endpoint returns status="unknown" rather than
        failing, so the sync API still starts on a stripped-down process.
    nlu: optional HestiaNLU instance, used only to surface classification
        cache counters on /health/modules.
    cors_origins: browser origins allowed to call this API cross-origin.
        This is a device-to-device sync endpoint, not something a web page
        should be calling, so it defaults to none (no wildcard) rather
        than the previous allow_origins=["*"], which let any web page in
        any tab make authenticated-looking requests to it.
    """
    application = FastAPI(
        title="Memory Sync API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    application.state.device_name = device_name
    application.state.api_key = api_key
    application.state.diagnostics = diagnostics
    application.state.nlu = nlu
    application.include_router(router)
    _install_middleware(application)
    return application




# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_memory(request: Request):
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory store not initialised")
    return memory


def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    api_key: Optional[str] = Query(default=None),
) -> None:
    """Guard for /sync/* routes. No-op if the app wasn't given an api_key
    (unauthenticated single-device/loopback use); otherwise requires an
    exact match via header or query param.
    """
    configured = getattr(request.app.state, "api_key", None)
    if not configured:
        return None
    supplied = x_api_key or api_key
    if supplied != configured:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

def _install_middleware(application: FastAPI) -> None:
    """
    Attach request logging and the catch-all exception handler.

    Same bug as the routes above: these were registered on the module-level
    `app`, so the instance main.py actually serves had neither. Called from
    create_app() so every instance gets both.
    """

    @application.middleware("http")
    async def request_logging(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "method=%s path=%s status=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request):
    """
    Liveness only: "is this process answering HTTP?".

    Deliberately does not probe any module — a client polling liveness
    every few seconds shouldn't trigger 20 module health probes. Use
    /health/modules for readiness. The registry version/fingerprint are
    included here (backlog #11) precisely because this is the cheap
    endpoint a client can poll to notice the intent set changed.
    """
    registry = registry_info()
    return HealthResponse(
        status="ok",
        device=request.app.state.device_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        registry_version=registry["version"],
        registry_fingerprint=registry["fingerprint"],
    )


@router.get(
    "/health/modules",
    response_model=ModulesHealthResponse,
    tags=["ops"],
)
async def health_modules(request: Request):
    """
    Aggregate every registered module's health into one JSON blob
    (backlog #19), so the web UI polls one endpoint instead of inferring
    module health from whichever feature happens to break first.

    Unauthenticated, like /health, and for the same reason: it reports
    subsystem *state* only — no memory contents, no user data, no config
    values. It is also read-only; nothing here can change module state.

    Never 500s. A missing Diagnostics (a stripped-down process, or the
    sync API started before the orchestrator was bound) reports
    status="unknown" with an empty module map, because a monitoring
    endpoint that throws is worse than one that admits it doesn't know.
    """
    diagnostics = getattr(request.app.state, "diagnostics", None)
    nlu = getattr(request.app.state, "nlu", None)
    timestamp = datetime.now(timezone.utc).isoformat()
    device = request.app.state.device_name

    nlu_cache = None
    if nlu is not None and hasattr(nlu, "cache_stats"):
        try:
            nlu_cache = nlu.cache_stats()
        except Exception:
            logger.debug("cache_stats() failed; omitting from health report.")

    if diagnostics is None:
        return ModulesHealthResponse(
            status="unknown",
            device=device,
            timestamp=timestamp,
            module_count=0,
            modules={},
            registry=registry_info(),
            nlu_cache=nlu_cache,
        )

    try:
        raw = diagnostics.module_status()
    except Exception:
        logger.exception("module_status() failed")
        raw = {}

    modules = {
        name: ModuleHealth(
            registered=bool(info.get("registered", True)),
            state=str(info.get("state", "unknown")),
            probe=info.get("probe"),
            detail=info.get("detail"),
        )
        for name, info in raw.items()
    }

    if not modules:
        status = "unknown"
    elif any(m.state in ("degraded", "error") for m in modules.values()):
        status = "degraded"
    else:
        # "unknown" per-module means the module exposes no probe, which is
        # the normal case for most of them — it isn't a fault, so it
        # doesn't drag the aggregate down.
        status = "ok"

    return ModulesHealthResponse(
        status=status,
        device=device,
        timestamp=timestamp,
        module_count=len(modules),
        modules=modules,
        registry=registry_info(),
        nlu_cache=nlu_cache,
    )


@router.get(
    "/sync/pull",
    response_model=PullResponse,
    tags=["sync"],
    dependencies=[Depends(require_api_key)],
)
async def sync_pull(
    since: Optional[str] = Query(
        default=None,
        description="ISO-8601 timestamp; return only interactions pushed after this point",
        example="2024-01-01T00:00:00+00:00",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    memory=Depends(get_memory),
):
    if since is not None:
        try:
            # Validate the caller supplied a parseable timestamp.
            datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="`since` must be a valid ISO-8601 timestamp",
            )

    try:
        if since:
            rows = memory.db.get_interactions_since(since, limit=limit)
        else:
            rows = memory.db.get_recent_interactions(limit=limit)

        return PullResponse(
            interactions=rows,
            timestamp=datetime.now(timezone.utc).isoformat(),
            count=len(rows),
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("sync_pull failed")
        raise HTTPException(status_code=500, detail="Failed to retrieve interactions")


@router.post(
    "/sync/push",
    response_model=PushResponse,
    tags=["sync"],
    dependencies=[Depends(require_api_key)],
)
async def sync_push(
    payload: PushRequest,
    memory=Depends(get_memory),
):
    if not payload.interactions:
        return PushResponse(status="ok", merged=0, skipped=0)

    try:
        existing = memory.db.get_recent_interactions(limit=5000)

        existing_keys: set[tuple[str, str, str]] = {
            (r["query"], r["response"], r["intent"])
            for r in existing
        }

        merged = 0
        skipped = 0

        for interaction in payload.interactions:
            key = (interaction.query, interaction.response, interaction.intent)

            if key in existing_keys:
                skipped += 1
                continue

            memory.db.push_interaction(
                interaction.query,
                interaction.response,
                interaction.intent,
            )
            existing_keys.add(key)   # prevent duplicates within the same batch
            merged += 1

        logger.info("sync_push complete merged=%d skipped=%d", merged, skipped)
        return PushResponse(status="ok", merged=merged, skipped=skipped)

    except HTTPException:
        raise
    except Exception:
        logger.exception("sync_push failed")
        raise HTTPException(status_code=500, detail="Failed to push interactions")


# ---------------------------------------------------------------------------
# Default app instance
# ---------------------------------------------------------------------------
# Kept for backwards compatibility (e.g. `uvicorn api:app`). main.py's
# start_sync_api builds its own instance via create_app(api_key=...), which
# now gets the same routes and middleware because both come from the
# router/_install_middleware rather than from decorators bound to this one
# object.
#
# This assignment MUST stay at the bottom of the file: create_app() calls
# include_router(router), so the router has to be fully populated (every
# @router.get/@router.post below has to have run) before this line executes.
# Constructing it at the top is exactly how an app with zero routes gets
# built without any error being raised.
app = create_app()
