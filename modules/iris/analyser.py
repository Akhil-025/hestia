# modules/iris/analyser.py

import base64
import json
import logging
import requests
from pathlib import Path
from PIL import Image, ExifTags
import io
import re
from datetime import datetime


def _extract_exif(image: "Image.Image") -> dict:
    """
    Pull date-taken, GPS coordinates, and camera make/model out of an
    already-open PIL Image's EXIF data (backlog #74). Returns an empty
    dict — never raises — for an image with no EXIF block at all (very
    common: screenshots, downloaded/re-saved images, most PNGs), a
    malformed one, or any of the individual tags being absent.
    """
    result: dict = {}
    try:
        exif = image.getexif()
        if not exif:
            return result
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

        date_str = tags.get("DateTimeOriginal") or tags.get("DateTime")
        if date_str:
            try:
                result["date_taken"] = datetime.strptime(
                    str(date_str), "%Y:%m:%d %H:%M:%S"
                ).isoformat()
            except ValueError:
                pass  # unexpected date format — skip rather than guess

        if tags.get("Make"):
            result["camera_make"] = str(tags["Make"]).strip()
        if tags.get("Model"):
            result["camera_model"] = str(tags["Model"]).strip()

        gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(exif, "get_ifd") else None
        if gps_info:
            lat = _gps_to_decimal(gps_info.get(2), gps_info.get(1))
            lon = _gps_to_decimal(gps_info.get(4), gps_info.get(3))
            if lat is not None:
                result["gps_lat"] = lat
            if lon is not None:
                result["gps_lon"] = lon
    except Exception as exc:  # pragma: no cover - defensive, EXIF is notoriously messy
        logging.getLogger(__name__).debug("EXIF extraction failed: %s", exc)
    return result


def _gps_to_decimal(coord, ref) -> "float | None":
    """
    Convert an EXIF GPS coordinate — a ((deg_num, deg_den), (min_num,
    min_den), (sec_num, sec_den)) tuple, or Pillow's newer IFDRational
    triple — plus a hemisphere reference ('N'/'S'/'E'/'W'), into decimal
    degrees. Returns None for anything that doesn't parse cleanly rather
    than guessing.
    """
    if not coord or not ref:
        return None
    try:
        degrees, minutes, seconds = (float(v) for v in coord)
        decimal = degrees + minutes / 60.0 + seconds / 3600.0
        if str(ref).upper() in ("S", "W"):
            decimal = -decimal
        return decimal
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _load_image_base64(file_path: Path, extract_exif: bool = False) -> tuple:
    """
    Open, downscale, and JPEG-encode an image file to base64 — the exact
    steps `analyse_file` already does per-image, extracted so
    `IrisEngine.describe_change` (backlog #76) can load two images the
    same way without duplicating the resize/encode logic. Returns
    (base64_str, exif_dict) — exif_dict is {} unless extract_exif=True.
    """
    with Image.open(file_path) as img:
        exif_data = _extract_exif(img) if extract_exif else {}
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8"), exif_data


class IrisAnalyser:
    def __init__(
        self,
        db,
        ollama_host: str,
        ollama_port: int,
        ollama_model: str = "llava:7b",
        embedder=None,     # ClipEmbedder | None — optional, degrades silently
        vector_index=None,  # ImageVectorIndex | None — optional, degrades silently
    ):
        self.db = db
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.ollama_model = ollama_model
        self.embedder = embedder
        self.vector_index = vector_index
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
            image_base64, exif_data = _load_image_base64(file_path, extract_exif=True)

            if exif_data:
                try:
                    self.db.update_file_exif(file_id, **exif_data)
                except Exception as exc:
                    self.logger.warning(f"Failed to store EXIF for file {file_id}: {exc}")

            # Semantic embedding is independent of the caption LLM call below
            # (different failure modes — a slow/unavailable Ollama vision
            # model shouldn't block CLIP indexing, and vice versa), so it
            # runs as its own best-effort step. Both `embedder` and
            # `vector_index` are optional and silently no-op if unavailable.
            self._embed_and_index(file_id, file_path)

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
            try:
                self.db.mark_file_error(file_id, str(e))  # retryable, unlike mark_file_processed
            except Exception:
                pass
            return False

    def _embed_and_index(self, file_id: int, file_path: Path) -> None:
        """
        Best-effort: compute a CLIP embedding for this image and upsert it
        into the vector index, keyed by Iris's own file_id. Never raises —
        a missing/unavailable embedder or vector_index (e.g. torch not
        installed) must not block caption analysis, which is the primary
        path this method is called from.
        """
        if self.embedder is None or self.vector_index is None:
            return
        try:
            vector = self.embedder.embed_image(file_path)
            if vector is None:
                return
            self.vector_index.upsert(file_id, vector)
        except Exception as e:
            self.logger.warning(f"[Iris] Embedding step failed for file {file_id}: {e}")

    def _send_to_ollama(self, images: "str | list", prompt: str) -> str:
        """
        *images* is a single base64 string (the normal single-photo
        analysis case) or a list of them — llava's ollama API accepts
        multiple images in one call, which is exactly what "describe what
        changed between these two photos" (backlog #76) needs: one call
        with both images, not two separate single-image analyses stitched
        together after the fact (which would lose any direct comparison
        the model could make between them).
        """
        image_list = images if isinstance(images, list) else [images]
        url = f"http://{self.ollama_host}:{self.ollama_port}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "images": image_list,
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
                else:
                    self.db.mark_queue_failed(queue_id, "Analysis returned False")
                    errors += 1

            except Exception as e:
                self.db.mark_queue_failed(queue_id, str(e))
                errors += 1

        return {"analysed": analysed, "errors": errors}

    def _parse_response(self, response: str):
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