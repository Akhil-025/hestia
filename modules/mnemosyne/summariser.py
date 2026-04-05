"""
modules/mnemosyne/summariser.py

"""
import logging
import json
from datetime import datetime
from .config import get_config

logger = logging.getLogger(__name__)

class Summariser:
    def __init__(self, engine, hestia_llm):
        self.engine = engine
        self.hestia_llm = hestia_llm
        self.config = get_config()

    def should_summarise(self) -> bool:
        cur = self.engine.db._conn.execute(
            "SELECT COUNT(*) FROM interaction_log WHERE summarised=0"
        )
        return cur.fetchone()[0] >= self.config.summarise_every_n

    def run(self) -> bool:
        interactions = self.engine.get_unsummarised()
        if len(interactions) < self.config.summarise_every_n:
            return False

        ids          = [i["id"] for i in interactions]
        period_start = interactions[0]["pushed_at"]
        period_end   = interactions[-1]["pushed_at"]
        texts        = [
            f"User: {i['user_text']}\nHestia: {i['hestia_response']}"
            for i in interactions
        ]
        joined = "\n".join(texts)

        prompt = (
            "Summarise the following conversation into a 3-5 sentence paragraph "
            "and a one-word topic label. "
            "Return only valid JSON: {\"summary\": \"...\", \"topic\": \"...\"}. "
            "No preamble.\n\n" + joined
        )

        try:
            llm_text = self.hestia_llm.generate(prompt)
            parsed   = json.loads(llm_text.strip())
            summary  = parsed.get("summary", "")
            topic    = parsed.get("topic", "General")
        except Exception:
            summary = ""
            topic   = "General"

        if not summary:
            return False

        self.engine.add_summary(period_start, period_end, summary, topic, len(interactions))
        self.engine.mark_summarised(ids)

        logger.info(f"Summarised {len(interactions)} interactions (topic: {topic})")
        return True