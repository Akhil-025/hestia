# api.py

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import time

logger = logging.getLogger(__name__)


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

def create_app(device_name: str = "laptop") -> FastAPI:
    application = FastAPI(
        title="Memory Sync API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    application.state.device_name = device_name
    return application


app = create_app()


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_memory(request: Request):
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory store not initialised")
    return memory


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
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


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request):
    return HealthResponse(
        status="ok",
        device=request.app.state.device_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/sync/pull", response_model=PullResponse, tags=["sync"])
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
            rows = memory.db.get_recent_interactions(limit=1000)
            rows = [r for r in rows if r.get("pushed_at", "") > since]
            rows = rows[:limit]
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


@app.post("/sync/push", response_model=PushResponse, tags=["sync"])
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