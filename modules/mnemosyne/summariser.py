# modules/mnemosyne/summariser.py

import logging
import json
from .config import get_config

logger = logging.getLogger(__name__)

class Summariser:
    def __init__(self, engine, hestia_llm):
        self.engine = engine
        self.hestia_llm = hestia_llm
        self.config = get_config()
        self._failure_count = 0
        self._skip_until = 0   # interaction count at which to retry

    def should_summarise(self) -> bool:
        cur = self.engine.db._conn.execute(
            "SELECT COUNT(*) FROM interaction_log WHERE summarised=0"
        )
        unsummarised = cur.fetchone()[0]

        if self._skip_until > 0:
            self._skip_until -= 1
            return False

        return unsummarised >= self.config.summarise_every_n

    def run(self) -> bool:
        interactions = self.engine.get_unsummarised()
        if len(interactions) < self.config.summarise_every_n:
            return False

        ids          = [i["id"] for i in interactions]
        period_start = interactions[0].get("pushed_at") or "unknown"
        period_end   = interactions[-1].get("pushed_at") or "unknown"
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

        summary = ""
        topic   = "General"

        try:
            llm_text = self.hestia_llm.generate(prompt)
            parsed   = json.loads(llm_text.strip())
            summary  = parsed.get("summary", "")
            topic    = parsed.get("topic", "General")
        except Exception as e:
            logger.warning("Summariser LLM failed: %s", e)

        if not summary:
            self._failure_count += 1
            # Back off: skip for N interactions after repeated failures
            backoff = min(self._failure_count * 10, 50)
            self._skip_until = backoff
            logger.warning(
                "Summariser: no summary generated (failure %d), "
                "skipping next %d interactions",
                self._failure_count, backoff
            )
            # Mark interactions with a placeholder so they don't pile up
            self.engine.db._conn.execute(
                "UPDATE interaction_log SET summarised=1 WHERE id IN (%s)"
                % ",".join("?" * len(ids)),
                ids
            )
            self.engine.db._conn.commit()
            return False

        self._failure_count = 0  # reset on success
        self.engine.add_summary(period_start, period_end, summary, topic, len(interactions))
        self.engine.mark_summarised(ids)

        logger.info("Summarised %d interactions (topic: %s)", len(interactions), topic)
        return True