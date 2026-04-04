import datetime
import platform

def register() -> dict:
    return {
        "intent": "get_system_info",
        "description": "Return system information like OS, uptime, Python version",
        "examples": [
            "what system are you running on",
            "tell me about this machine",
            "what OS is this",
            "system info",
        ],
        "execute": execute
    }

def execute(entities: dict, memory, raw_query: str = ""):
    """Return a brief system info string."""
    os_info = platform.system() + " " + platform.release()
    py_version = platform.python_version()
    now = datetime.datetime.now().strftime("%I:%M %p, %A %B %d")
    return (
        f"Running on {os_info}, Python {py_version}. "
        f"Current time is {now}."
    )
