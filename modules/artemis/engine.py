# artemis/engine.py

from modules.base import BaseModule  
from .tracker import ArtemisTracker

class ArtemisEngine(BaseModule):     
    name = "artemis"

    def __init__(self):
        self.tracker = ArtemisTracker()

    def can_handle(self, intent: str) -> bool:
        return intent in {
            "add_habit", "complete_habit", "list_habits",
            "add_goal", "update_goal", "list_goals", "productivity_summary"
        }

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        response = ""
        data = {}
        confidence = 0.9
        if intent == "add_habit":
            name = entities.get("name", "").strip()
            if name:
                self.tracker.add_habit(name)
                response = f"Added habit '{name}'."
            else:
                response = "Please specify a habit name."
        elif intent == "complete_habit":
            name = entities.get("name", "").strip()
            if name:
                self.tracker.complete_habit(name)
                habits = self.tracker.get_habits()
                streak = habits.get(name, {}).get("streak", 0)
                response = f"Marked '{name}' complete. Current streak: {streak} days."
            else:
                response = "Please specify a habit name."
        elif intent == "list_habits":
            habits = self.tracker.get_habits()
            if not habits:
                response = "You have no habits tracked."
            else:
                parts = [f"{h} ({v['streak']}🔥)" for h, v in habits.items()]
                response = f"You have {len(habits)} habits: " + ", ".join(parts)
            data = habits
        elif intent == "add_goal":
            name = entities.get("name", "").strip()
            if name:
                self.tracker.add_goal(name)
                response = f"Goal '{name}' added."
            else:
                response = "Please specify a goal name."
        elif intent == "update_goal":
            name = entities.get("name", "").strip()
            progress = entities.get("progress")
            if name and progress is not None:
                try:
                    progress = float(progress)
                except Exception:
                    progress = 0.0
                self.tracker.update_goal(name, progress / 100 if progress > 1 else progress)
                pct = int((progress / 100 if progress > 1 else progress) * 100)
                response = f"Goal '{name}' is now {pct}% complete."
            else:
                response = "Please specify a goal name and progress."
        elif intent == "list_goals":
            goals = self.tracker.get_goals()
            if not goals:
                response = "You have no active goals."
            else:
                parts = [f"{g} ({int(v['progress']*100)}%)" for g, v in goals.items() if v.get("status") == "active"]
                response = "Active goals: " + ", ".join(parts)
            data = goals
        elif intent == "productivity_summary":
            habits = self.tracker.get_habits()
            goals = self.tracker.get_goals()
            total_habits = len(habits)
            avg_streak = round(sum(h["streak"] for h in habits.values()) / total_habits, 1) if total_habits else 0.0
            goals_progress = {g: v["progress"] for g, v in goals.items()}
            insights = self.analyze()

            response = (
                f"Avg streak: {insights['avg_streak']} days. "
                f"Strong: {', '.join(insights['strong_habits']) or 'none'}. "
                f"Needs focus: {', '.join(insights['weak_habits']) or 'none'}."
                f" Goals progress: " + ", ".join(f"{g} ({int(p*100)}%)" for g, p in goals_progress.items())
            )
            data = {"avg_streak": avg_streak, "habits": habits, "goals": goals}
            confidence = 0.8
        else:
            response = "Artemis can't handle that request."
            confidence = 0.5
        return {"response": response, "data": data, "confidence": confidence}
    

    def get_context(self) -> dict:               # ADD
        """Expose current habit/goal state for Hecate and secondary enrichment."""
        try:
            habits = self.tracker.get_habits()
            goals  = self.tracker.get_goals()
            return {
                "habit_count":    len(habits),
                "active_goals":   [k for k, v in goals.items() if v.get("status") == "active"],
                "avg_streak":     self.analyze().get("avg_streak", 0),
            }
        except Exception:
            return {}
    
    def analyze(self):
        habits = self.tracker.get_habits()
        goals = self.tracker.get_goals()

        insights = {
            "weak_habits": [],
            "strong_habits": [],
            "avg_streak": 0,
        }

        if habits:
            avg = sum(h["streak"] for h in habits.values()) / len(habits)
            insights["avg_streak"] = round(avg, 1)

            for name, h in habits.items():
                if h["streak"] < avg:
                    insights["weak_habits"].append(name)
                else:
                    insights["strong_habits"].append(name)

        return insights
    
    def suggest_next_action(self):
        insights = self.analyze()

        if insights["weak_habits"]:
            return f"You should focus on '{insights['weak_habits'][0]}' next."

        return "You're on track. Continue your habits."
