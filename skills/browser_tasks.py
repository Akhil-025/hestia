import sys

_browser_agent = None

def set_browser_agent(agent) -> None:
    """Inject browser agent reference into this skill module."""
    global _browser_agent
    _browser_agent = agent

def register() -> dict:
    return {
        "intent": "browser_action",
        "description": "Open URLs, search the web deeply, check flight status, fill forms",
        "examples": [
            "open youtube",
            "browse to google.com",
            "check flight AI302 status",
            "deep search latest python features",
            "what does the BBC homepage say",
        ],
        "execute": execute
    }

def execute(entities: dict, memory, raw_query: str = ""):
    """Route browser action based on entities."""
    if _browser_agent is None:
        return "Browser automation is not enabled."

    action = entities.get("action", "").lower()
    url = entities.get("url", "")
    query = entities.get("query") or entities.get("topic") or raw_query
    flight = entities.get("flight_number", "")

    # Route by action type
    if flight:
        return _browser_agent.check_flight_status(flight)

    if action in ("open", "browse", "navigate", "go to") and url:
        return _browser_agent.open_url(url)

    # Only allow explicit deep search
    if action in ("deep search",) and query:
        return _browser_agent.search_web(query)

    if url:
        return _browser_agent.open_url(url)

    if query:
        return _browser_agent.search_web(query)

    return "I'm not sure what browser action to take. Try saying 'search for X' or 'open youtube.com'."
