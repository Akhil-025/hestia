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
    c = memory.conn.cursor()
    if since:
        c.execute("SELECT id, timestamp, query, response, intent FROM interactions WHERE timestamp > ? ORDER BY timestamp ASC", (since,))
    else:
        c.execute("SELECT id, timestamp, query, response, intent FROM interactions ORDER BY timestamp DESC LIMIT 100")
    rows = c.fetchall()
    interactions = [
        {"id": r[0], "timestamp": r[1], "query": r[2], "response": r[3], "intent": r[4]} for r in rows
    ]
    now = datetime.now().isoformat()
    return {"interactions": interactions, "timestamp": now}

@app.post("/sync/push")
async def sync_push(request: Request):
    memory = request.app.state.memory
    data = await request.json()
    interactions = data.get("interactions", [])
    merged = 0
    c = memory.conn.cursor()
    for row in interactions:
        # Check for duplicates using timestamp and query
        c.execute("SELECT 1 FROM interactions WHERE timestamp=? AND query=?", 
                  (row['timestamp'], row['query']))
        if c.fetchone():
            continue
        # Insert without id - let SQLite AUTOINCREMENT assign it
        c.execute(
            "INSERT INTO interactions (timestamp, query, response, intent) VALUES (?, ?, ?, ?)",
            (row['timestamp'], row['query'], row['response'], row['intent'])
        )
        merged += 1
    memory.conn.commit()
    return {"status": "ok", "merged": merged}