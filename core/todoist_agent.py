"""
core/todoist_agent.py

Thin Todoist client for backlog #91 (task prioritisation and sorting).

Targets Todoist's unified **API v1** (``https://api.todoist.com/api/v1``).
The older ``/rest/v2`` endpoints have been shut down, so nothing here uses
them. Open tasks are fetched with ``GET /tasks`` and filtered/sorted on our
side from each task's own ``due`` field, so we don't depend on Todoist's
filter-query syntax.

The HTTP layer is a single injectable callable (``transport``) so the whole
agent can be tested with a fake. The default transport uses ``urllib`` from
the standard library, so there is no new dependency.

Nothing is read or written until a token is configured; ``is_ready()`` is
``False`` without one and callers should say so rather than guess.

Status: written against Todoist's published v1 shapes, tested only against a
fake transport. It has not been run against a real account.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.todoist.com/api/v1"
TOKEN_ENV_VAR = "TODOIST_API_TOKEN"
_MAX_PAGES = 10          # 10 pages x 50 = plenty for a personal task list
_DEFAULT_TIMEOUT = 10.0

# (method, url, headers, query params, JSON body, timeout) -> (status, parsed JSON or None)
Transport = Callable[[str, str, dict, Optional[dict], Optional[dict], float], tuple]


class TodoistError(Exception):
    """Raised for any failure talking to Todoist (network, auth, bad reply)."""


@dataclass(frozen=True)
class TodoistTask:
    id: str
    content: str
    priority: int = 1                 # Todoist API: 4 = urgent (shown as "p1"), 1 = normal
    due_date: Optional[date] = None
    due_string: str = ""
    project_id: str = ""
    labels: tuple = field(default_factory=tuple)

    @property
    def ui_priority(self) -> int:
        """The number a person sees in the Todoist app: p1 (urgent) .. p4."""
        return 5 - max(1, min(4, self.priority))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "priority": self.priority,
            "ui_priority": self.ui_priority,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "due_string": self.due_string,
            "project_id": self.project_id,
            "labels": list(self.labels),
        }

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "TodoistTask":
        due = raw.get("due") or {}
        due_date = _parse_due_date(due.get("date")) if isinstance(due, dict) else None
        try:
            priority = int(raw.get("priority") or 1)
        except (TypeError, ValueError):
            priority = 1
        return cls(
            id=str(raw.get("id", "")),
            content=str(raw.get("content") or "").strip(),
            priority=max(1, min(4, priority)),
            due_date=due_date,
            due_string=str(due.get("string") or "") if isinstance(due, dict) else "",
            project_id=str(raw.get("project_id") or ""),
            labels=tuple(str(x) for x in (raw.get("labels") or [])),
        )


def _parse_due_date(value: Any) -> Optional[date]:
    """Todoist sends ``YYYY-MM-DD`` or a full ISO datetime; keep just the date."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None


def _urllib_transport(
    method: str, url: str, headers: dict, params: Optional[dict],
    body: Optional[dict], timeout: float,
) -> tuple:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    hdrs = dict(headers)
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https base)
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TodoistError(f"Network error: {exc}") from exc
    if not raw:
        return status, None
    try:
        return status, json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise TodoistError("Todoist sent a reply that isn't valid JSON.") from exc


class TodoistAgent:
    """Read, add and complete Todoist tasks."""

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: Optional[Transport] = None,
    ) -> None:
        self._token = (token or os.environ.get(TOKEN_ENV_VAR) or "").strip()
        self._base = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._transport: Transport = transport or _urllib_transport

    def is_ready(self) -> bool:
        return bool(self._token)

    def _call(
        self, method: str, path: str, *,
        params: Optional[dict] = None, body: Optional[dict] = None,
    ) -> Any:
        if not self._token:
            raise TodoistError("No Todoist API token is configured.")
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        status, payload = self._transport(
            method, f"{self._base}{path}", headers, params, body, self._timeout
        )
        if status in (401, 403):
            raise TodoistError("Todoist rejected the API token.")
        if status == 404:
            raise TodoistError("Todoist couldn't find that task.")
        if status == 429:
            raise TodoistError("Todoist is rate-limiting requests; try again shortly.")
        if not 200 <= int(status) < 300:
            raise TodoistError(f"Todoist returned HTTP {status}.")
        return payload

    def list_tasks(self, project_id: Optional[str] = None) -> list[TodoistTask]:
        """All open tasks (optionally one project), following pagination."""
        params: dict[str, Any] = {"limit": 200}
        if project_id:
            params["project_id"] = project_id
        tasks: list[TodoistTask] = []
        for _ in range(_MAX_PAGES):
            payload = self._call("GET", "/tasks", params=dict(params))
            # v1 wraps lists as {"results": [...], "next_cursor": ...}; accept a bare list too.
            if isinstance(payload, dict):
                items = payload.get("results") or []
                cursor = payload.get("next_cursor")
            elif isinstance(payload, list):
                items, cursor = payload, None
            else:
                raise TodoistError("Unexpected reply shape from Todoist.")
            tasks.extend(TodoistTask.from_api(r) for r in items if isinstance(r, dict))
            if not cursor:
                break
            params["cursor"] = cursor
        return [t for t in tasks if t.id and t.content]

    def add_task(
        self, content: str, *, due_string: str = "", priority: Optional[int] = None,
        project_id: str = "",
    ) -> TodoistTask:
        """Create a task. ``priority`` is the API value (4 = urgent)."""
        content = (content or "").strip()
        if not content:
            raise ValueError("content must be non-empty.")
        body: dict[str, Any] = {"content": content}
        if due_string:
            body["due_string"] = due_string
        if priority is not None:
            body["priority"] = max(1, min(4, int(priority)))
        if project_id:
            body["project_id"] = project_id
        payload = self._call("POST", "/tasks", body=body)
        if not isinstance(payload, dict):
            raise TodoistError("Todoist didn't return the created task.")
        return TodoistTask.from_api(payload)

    def complete_task(self, task_id: str) -> None:
        """Mark a task done (``POST /tasks/{id}/close``)."""
        if not task_id:
            raise ValueError("task_id must be non-empty.")
        self._call("POST", f"/tasks/{urllib.parse.quote(str(task_id), safe='')}/close")


# ---------------------------------------------------------------------------
# Pure helpers: ranking and matching
# ---------------------------------------------------------------------------

def rank_tasks(tasks: list[TodoistTask], today: date) -> list[TodoistTask]:
    """
    Most-important-first ordering.

    Overdue tasks first (oldest first), then due today, then future-dated by
    date, then undated; inside each group higher Todoist priority wins, then
    the task text for a stable order.
    """
    def key(t: TodoistTask) -> tuple:
        if t.due_date is None:
            bucket, when = 3, 0
        elif t.due_date < today:
            bucket, when = 0, t.due_date.toordinal()
        elif t.due_date == today:
            bucket, when = 1, 0
        else:
            bucket, when = 2, t.due_date.toordinal()
        return (bucket, when, -t.priority, t.content.lower())
    return sorted(tasks, key=key)


def select_due(tasks: list[TodoistTask], today: date, scope: str) -> list[TodoistTask]:
    """Filter by ``today`` (today + overdue), ``overdue``, ``week`` or ``all``."""
    scope = (scope or "today").strip().lower()
    if scope in ("all", "everything", "open"):
        return list(tasks)
    if scope == "overdue":
        return [t for t in tasks if t.due_date and t.due_date < today]
    if scope in ("week", "this week", "7 days"):
        return [t for t in tasks if t.due_date and (t.due_date - today).days <= 7]
    return [t for t in tasks if t.due_date and t.due_date <= today]


def find_matches(tasks: list[TodoistTask], query: str) -> list[TodoistTask]:
    """Tasks whose text matches *query*: exact match wins, else every word present."""
    q = (query or "").strip().lower()
    if not q:
        return []
    exact = [t for t in tasks if t.content.lower() == q]
    if exact:
        return exact
    words = [w for w in q.split() if w]
    return [t for t in tasks if all(w in t.content.lower() for w in words)]
