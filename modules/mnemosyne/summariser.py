"""
modules/mnemosyne/summariser.py

"""
import logging
import json
from datetime import datetime
from .config import get_config

logger = logging.getLogger(__name__)

class Summariser:
    def __init__(self, db, vector_store, hestia_llm):
        self.db = db
        self.vector_store = vector_store
        self.hestia_llm = hestia_llm
        self.config = get_config()

    def should_summarise(self) -> bool:
        cur = self.db._conn.execute(
            "SELECT COUNT(*) FROM interaction_log WHERE summarised=0"
        )
        return cur.fetchone()[0] >= self.config.summarise_every_n

    def run(self) -> bool:
        interactions = self.db.get_unsummarised()
        if len(interactions) < self.config.summarise_every_n:
            return False
        ids = [i["id"] for i in interactions]
        period_start = interactions[0]["pushed_at"]
        period_end = interactions[-1]["pushed_at"]
        texts = [f"User: {i['user_text']}\nHestia: {i['hestia_response']}" for i in interactions]
        joined = "\n".join(texts)
        prompt = (
            "Summarise the following conversation into a 3-5 sentence paragraph and a one-word topic label. "
            "Return only valid JSON in the format: {\"summary\": \"...\", \"topic\": \"...\"}. "
            "No preamble, no explanation.\n\n" + joined
        )
        try:
            llm_text = self.hestia_llm.generate(prompt)

            parsed = json.loads(llm_text.strip())
            summary = parsed.get("summary", "")
            topic = parsed.get("topic", "General")
        except Exception:
            summary = ""
            topic = "General"

        if not summary:
            return False
        summary_id = self.db.add_summary(period_start, period_end, summary, topic, len(interactions))
        # Embed and add to vector store
        if self.vector_store:
            metadata = {"type": "summary", "created_at": datetime.utcnow().isoformat(), "topic": topic}
            self.vector_store.add(summary, metadata, doc_id=str(summary_id))
        self.db.mark_summarised(ids)
        logger.info(f"Summarised {len(interactions)} interactions (topic: {topic})")
        return True
