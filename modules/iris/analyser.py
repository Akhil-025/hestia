import base64
import json
import logging
import requests
from pathlib import Path
from PIL import Image
import io
import re

class IrisAnalyser:
    def __init__(self, db, ollama_host: str, ollama_port: int, ollama_model: str = "llava:7b"):
        self.db = db
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.ollama_model = ollama_model
        self.logger = logging.getLogger(__name__)

    def analyse_file(self, file_id: int) -> bool:
        try:
            file_record = self.db.get_file(file_id)
            if not file_record:
                self.logger.error(f"File ID {file_id} not found in DB.")
                return False
            file_path = Path(file_record.get("file_path"))
            file_type = file_record.get("file_type", "")
            if not file_path.exists():
                self.logger.error(f"File not found on disk: {file_path}")
                return False
            if file_type != "image":
                return True
  
            UNSUPPORTED_EXTS = {'.heic', '.heif', '.raw', '.nef', '.cr2',
                                '.arw', '.dng', '.orf', '.sr2'}
            if file_path.suffix.lower() in UNSUPPORTED_EXTS:
                self.logger.info(f"Skipping unsupported format: {file_path.name}")
                return True
            self.logger.info(f"Analysing image: {file_path}")
            with Image.open(file_path) as img:
                # Convert RGBA → RGB (fix crash)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                img.thumbnail((1024, 1024))

                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            prompt = (
                "You are an image description assistant. "
                "You MUST respond using ONLY this exact format with no other text:\n\n"
                "CAPTION: A clear one-sentence description of the image.\n"
                "TAGS: tag1, tag2, tag3, tag4, tag5\n"
                "MOOD: oneword\n\n"
                "Example response:\n"
                "CAPTION: A group of friends laughing at a birthday party.\n"
                "TAGS: people, party, birthday, friends, celebration\n"
                "MOOD: joyful\n\n"
                "Now describe the image following this format exactly."
            )
            response = self._send_to_ollama(image_base64, prompt)
            if not response:
                raise RuntimeError("Empty response from Ollama")
            # Parse and update DB, always mark processed, handle errors
            try:
                caption, tags, mood = self._parse_response(response)

                # Ensure safe defaults
                caption = caption or "Unlabeled photo"
                tags = tags or "photo"
                mood = mood or "neutral"

                # Convert tags to JSON string
                import json
                tags_json = json.dumps([t.strip() for t in tags.split(",") if t.strip()])

                # Write to DB
                self.db.update_file_analysis(
                    file_id,
                    caption,
                    tags_json,
                    None,       # objects (not used yet)
                    mood,
                    False,      # is_sensitive
                    None        # blur_score
                )

                # Mark as processed
                self.db.mark_file_processed(file_id)

                return True

            except Exception as e:
                self.logger.error(f"Parse/update failed for file {file_id}: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Error analysing file {file_id}: {e}")
            return False

    def _send_to_ollama(self, image_base64: str, prompt: str) -> str:
        url = f"http://{self.ollama_host}:{self.ollama_port}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            self.logger.error(f"Ollama API error: {e}")
            return ""

    def run_batch(self, limit: int = 10) -> dict:
        analysed = 0
        skipped = 0
        errors = 0

        for _ in range(limit):
            item = self.db.get_next_queued()
            if not item:
                break

            queue_id = item["id"]
            file_id = item["file_id"]

            self.db.mark_queue_processing(queue_id)

            try:
                result = self.analyse_file(file_id)

                if result is True:
                    self.db.mark_queue_done(queue_id)
                    analysed += 1
                elif result == "skipped":
                    self.db.mark_queue_done(queue_id)
                    skipped += 1
                else:
                    self.db.mark_queue_failed(queue_id, "Analysis returned False")
                    errors += 1

            except Exception as e:
                self.db.mark_queue_failed(queue_id, str(e))
                errors += 1

        return {"analysed": analysed, "skipped": skipped, "errors": errors}

    def _parse_response(self, response: str):
        import re
        caption = tags = mood = None

        # Try structured parse first
        cap_match  = re.search(r"CAPTION:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        tag_match  = re.search(r"TAGS:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        mood_match = re.search(r"MOOD:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)

        if cap_match:
            caption = cap_match.group(1).strip()
        if tag_match:
            tags = tag_match.group(1).strip()
        if mood_match:
            mood = mood_match.group(1).strip()

        # Fallback: if structured parse failed, use free-form response as caption
        if not caption and response.strip():
            lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
            # Use first non-empty line as caption (truncated to 200 chars)
            caption = lines[0][:120] if lines else response[:200]
            # Extract last word of response as mood fallback
            words = response.split()
            mood = mood or (words[-1].strip('.,!?').lower() if words else "neutral")
            # Use all lines joined as tags fallback
            tags = tags or "photo"

        return caption, tags, mood