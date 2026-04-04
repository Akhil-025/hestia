import requests
from bs4 import BeautifulSoup

from core.ollama_client import generate

def register():
    return {
        "intent": "search_web",
        "description": "Search the web for general information",
        "examples": [
            "search python asyncio",
            "look up finite element method",
            "what is turbulence modeling"
        ],
        "execute": execute
    }

def execute(entities: dict, memory, raw_query: str = "") -> str:
    query = raw_query.strip()

    if not query:
        return "What should I search for?"

    # clean command words
    import re
    query = re.sub(r"^(search|look up|google)\s+", "", query, flags=re.IGNORECASE).strip()

    url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        results = soup.select("a.result__a")

        if not results:
            return f"No results found for '{query}'."

        # return top 3 results
        snippets = []

        for r in results[:3]:
            title = r.get_text()
            parent = r.find_parent("div")
            snippet = parent.get_text() if parent else title
            snippets.append(snippet[:200])

        results_text = "\n".join(snippets)

        prompt = f"""
        Answer the user's question clearly and accurately.

        User query: {query}

        Search results:
        {results_text}

        Instructions:
        - Give a correct technical explanation
        - Do NOT invent APIs, classes, or functions
        - If unsure about specifics, stay general
        - Prefer correctness over detail
        - Keep answer concise (2–4 sentences)
        """

        try:
            summary = generate(prompt, model="mistral")
            return summary
        except:
            return results_text

    except Exception as e:
        print(f"[Search] error: {e}")
        return "Search failed."