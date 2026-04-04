"""
modules/mnemosyne/summariser.py

Summariser for Mnemosyne: compresses interactions into summaries and stores in DB/vector store.
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
        unsummarised = self.db.get_unsummarised()
        return len(unsummarised) >= self.config.summarise_every_n

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
        llm_result = self.hestia_llm.generate(prompt)
        if isinstance(llm_result, dict):
            llm_text = llm_result.get("text", "")
        else:
            llm_text = llm_result
        try:
            parsed = json.loads(llm_text)
            summary = parsed.get("summary", "")
            topic = parsed.get("topic", "General")
        except Exception:
            # Fallback: try to extract summary/topic heuristically
            summary = llm_text.strip()
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
