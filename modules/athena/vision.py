#athena/vision.py

import base64
import logging
import requests
from io import BytesIO

logger = logging.getLogger(__name__)


class VisionModel:
    def __init__(self, model="llava"):
        self.model = model
        self.url = "http://127.0.0.1:11434/api/generate"

    def _encode_image(self, image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def describe(self, image) -> str:
        try:
            img_b64 = self._encode_image(image)

            payload = {
                "model": self.model,
                "prompt": "Describe this engineering diagram clearly and concisely.",
                "images": [img_b64],
                "stream": False
            }

            response = requests.post(self.url, json=payload, timeout=30)
            response.raise_for_status()

            return response.json().get("response", "").strip()

        except Exception as e:
            logger.warning("[VisionModel] describe() failed: %s", e, exc_info=True)
            return ""