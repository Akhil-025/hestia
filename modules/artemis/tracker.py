# artemis/tracker.py

import os
import json
from datetime import datetime, timedelta

class ArtemisTracker:
    def __init__(self, path="data/artemis_state.json"):
        self.path = path
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"habits": {}, "goals": {}}, f)

    def load(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: dict):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_habit(self, name: str):
        data = self.load()
        if name not in data["habits"]:
            data["habits"][name] = {"streak": 0, "last_done": ""}
            self.save(data)

    def complete_habit(self, name: str):
        data = self.load()
        today = datetime.now().date().isoformat()
        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
        habit = data["habits"].get(name)
        if not habit:
            return
        last_done = habit.get("last_done", "")
        if last_done == today:
            pass  # already done today
        elif last_done == yesterday:
            habit["streak"] += 1
            habit["last_done"] = today
        else:
            habit["streak"] = 1
            habit["last_done"] = today
        self.save(data)

    def get_habits(self) -> dict:
        data = self.load()
        return data.get("habits", {})

    def add_goal(self, name: str):
        data = self.load()
        if name not in data["goals"]:
            data["goals"][name] = {"progress": 0.0, "status": "active"}
            self.save(data)

    def update_goal(self, name: str, progress: float):
        data = self.load()
        goal = data["goals"].get(name)
        if goal:
            goal["progress"] = max(0.0, min(1.0, progress))
            self.save(data)

    def get_goals(self) -> dict:
        data = self.load()
        return data.get("goals", {})
