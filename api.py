# api.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime

app = FastAPI()

@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "device": "laptop"}

@app.get("/sync/pull")
async def sync_pull(request: Request, since: Optional[str] = None):
    memory = request.app.state.memory

    try:
        if since:
            rows = memory.db.get_recent_interactions(1000)
            rows = [r for r in rows if r.get("timestamp") > since]
        else:
            rows = memory.db.get_recent_interactions(100)

        now = datetime.now().isoformat()
        return {"interactions": rows, "timestamp": now}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/sync/push")
async def sync_push(request: Request):
    memory = request.app.state.memory
    data = await request.json()
    interactions = data.get("interactions", [])

    merged = 0

    try:
        existing = memory.db.get_recent_interactions(5000)

        existing_keys = {
            (r["query"], r["response"], r["intent"])
            for r in existing
        }

        for row in interactions:
            key = (row["query"], row["response"], row["intent"])

            if key in existing_keys:
                continue

            memory.db.push_interaction(
                row["query"],
                row["response"],
                row["intent"]
            )
            merged += 1

        return {"status": "ok", "merged": merged}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )