"""
modules/athena/pdf_processor.py

PDF text extraction (with OCR fallback) and semantic chunking.
Fixed for Hestia integration: config import is now relative.
"""

import io
import logging
import os
import platform
import re
import shutil
from typing import Any, Dict, List, Optional

import fitz  # pymupdf
import numpy as np
from PIL import Image
from modules.athena.vision import VisionModel
from modules.athena.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCR_FAILED_MARKER = "[OCR_FAILED_PAGE]"

SUPPORTED_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".pptx", ".txt", ".md", ".epub",
})

_OCR_DPI                    = 300
_BLANK_PAGE_STD_THRESHOLD   = 2.0
_MIN_OCR_WORDS              = 5
_MIN_TEXT_CHARS_FOR_DIGITAL = 200

_MIN_PARAGRAPH_CHARS = 20
_MAX_HEADING_CHARS  = 120


# ---------------------------------------------------------------------------
# Tesseract configuration
# ---------------------------------------------------------------------------

def _configure_tesseract() -> None:
    import pytesseract

    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path
        return

    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        return

    if platform.system() == "Windows":
        win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(win_default):
            pytesseract.pytesseract.tesseract_cmd = win_default
            return

    logger.warning(
        "Tesseract not found — OCR will fail. "
        "Install Tesseract or set the TESSERACT_CMD environment variable."
    )


try:
    _configure_tesseract()
except Exception:
    pass  # pytesseract may not be installed; OCR will fail gracefully


# ---------------------------------------------------------------------------
# PDFProcessor
# ---------------------------------------------------------------------------

class PDFProcessor:

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        import torch

        config = get_config()
        self.chunk_size: int    = chunk_size    or config.chunk_size
        self.chunk_overlap: int = chunk_overlap or config.chunk_overlap
        self.use_gpu: bool      = torch.cuda.is_available()
        if self.use_gpu:
            torch.cuda.set_device(0)
        self._ocr_reader: Any   = None
        self.vision_model: VisionModel = VisionModel()

        logger.info(
            "PDFProcessor: chunk_size=%d, overlap=%d, GPU=%s",
            self.chunk_size, self.chunk_overlap, self.use_gpu,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        pages = self._extract_pymupdf(file_path)
        has_digital_text = any(
            len(p["text"]) > _MIN_TEXT_CHARS_FOR_DIGITAL for p in pages
        )
        if not has_digital_text:
            try:
                pages = self._extract_ocr(file_path)
            except Exception:
                logger.warning("OCR failed for %s — using sparse digital text", file_path)
        return pages

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        pages = self.extract_text_from_pdf(file_path)
        all_chunks: List[Dict[str, Any]] = []

        for page in pages:
            if page["text"] == OCR_FAILED_MARKER:
                continue
            chunks = self.semantic_chunking(page["text"])
            for idx, chunk_text in enumerate(chunks, start=1):
                if len(chunk_text.strip()) < 20:
                    continue
                all_chunks.append({
                    "text":         chunk_text,
                    "file_name":    page["file_name"],
                    "file_path":    page["file_path"],
                    "page_number":  page["page_number"],
                    "chunk_number": idx,
                    "total_chunks": len(chunks),
                    "total_pages":  page["total_pages"],
                })

        logger.info("Created %d chunks for %s", len(all_chunks), os.path.basename(file_path))
        return all_chunks

    # ------------------------------------------------------------------
    # Extraction backends
    # ------------------------------------------------------------------

    def _extract_pymupdf(self, file_path: str) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        with fitz.open(file_path) as doc:
            total = len(doc)
            for i, page in enumerate(doc, start=1):
                text = self.clean_text(page.get_text("text") or "")
                if text:
                    pages.append(self._make_page(text, i, file_path, total))
        return pages

    def _extract_ocr(self, file_path: str) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        with fitz.open(file_path) as doc:
            total = len(doc)
            for i, page in enumerate(doc, start=1):
                pix = page.get_pixmap(dpi=_OCR_DPI)
                pil_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")

                img_np = np.array(pil_img)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
                img_np = (img_np * 255).astype("uint8")
                img_np = (img_np > 150).astype("uint8") * 255
                pil_img = Image.fromarray(img_np)

                if self._is_blank_page(pil_img):
                    continue

                cleaned = self._tesseract_ocr(pil_img)

                if self.use_gpu:
                    gpu_text = self._easyocr_fallback(pil_img, page_number=i)
                    if gpu_text:
                        cleaned = gpu_text

                vision_text = self._vision_describe(pil_img)
                combined = cleaned + "\n\n[VISUAL]\n" + vision_text
                pages.append(self._make_page(combined, i, file_path, total))

                print(f"[OCR DEBUG] Page {i}:", cleaned[:200])

        return pages

    def _tesseract_ocr(self, image: Image.Image) -> str:
        import pytesseract
        try:
            raw =   pytesseract.image_to_string(
                        image,
                        config="--oem 3 --psm 4 -l eng"
                    )
            return self.clean_text(raw)
        except Exception:
            logger.debug("Tesseract failed", exc_info=True)
            return ""

    def _easyocr_fallback(self, image: Image.Image, page_number: int) -> str:
        if self._ocr_reader is None:
            import easyocr
            logger.info("Loading EasyOCR on GPU …")
            self._ocr_reader = easyocr.Reader(["en"], gpu=True)
        try:
            img_np = np.array(image)
            results = self._ocr_reader.readtext(img_np, detail=0, batch_size=8)
            return self.clean_text(" ".join(results))
        except Exception:
            logger.warning("EasyOCR failed on page %d", page_number, exc_info=True)
            return ""
    
    def _vision_describe(self, image):
        try:
            return self.vision_model.describe(image)
        except:
            return ""

    # ------------------------------------------------------------------
    # Text cleaning & chunking
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def semantic_chunking(self, text: str) -> List[str]:
        heading_re = re.compile(
            r"^(?:[A-Z][A-Z\s]{3,}"
            r"|(?:\d+\.)+\s+\w"
            r"|[IVXLC]+\.\s+\w"
            r").{0,80}$",
            re.MULTILINE,
        )

        sentences = re.split(r'[.?!]\s+', text)

        lines = [s.strip() for s in sentences if len(s.strip()) > 20]

        paragraphs = []
        buffer = ""

        for line in lines:
            TARGET_SIZE = 300

            if len(buffer) < TARGET_SIZE:
                buffer += " " + line
            else:
                paragraphs.append(buffer.strip())
                buffer = line

        if buffer:
            paragraphs.append(buffer.strip())

        chunks: List[str] = []
        current_chunk     = ""
        current_heading   = ""

        for para in paragraphs:
            if heading_re.match(para) and len(para) < _MAX_HEADING_CHARS:
                current_heading = para
                continue

            candidate = f"{current_heading}\n{para}" if current_heading else para

            if not current_chunk:
                current_chunk = candidate
            elif len(current_chunk) + len(candidate) < self.chunk_size:
                current_chunk += "\n\n" + candidate
            else:
                chunks.append(current_chunk.strip())
                last_para = current_chunk.split("\n\n")[-1]
                current_chunk = (
                    f"{current_heading}\n{last_para}\n\n{candidate}"
                    if current_heading
                    else f"{last_para}\n\n{candidate}"
                )

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_page(text: str, page_number: int, file_path: str, total_pages: int) -> Dict[str, Any]:
        return {
            "text":        text,
            "page_number": page_number,
            "file_name":   os.path.basename(file_path),
            "file_path":   file_path,
            "total_pages": total_pages,
        }

    @staticmethod
    def _is_blank_page(pil_img: Image.Image) -> bool:
        return float(np.array(pil_img).std()) < _BLANK_PAGE_STD_THRESHOLD


# ---------------------------------------------------------------------------
# File-discovery utilities
# ---------------------------------------------------------------------------

def get_supported_files(data_dir: Optional[str] = None) -> List[Dict[str, str]]:
    if data_dir is None:
        data_dir = str(get_config().data_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        logger.warning("Created data directory: %s", data_dir)
        return []

    files: List[Dict[str, str]] = []
    for root, _, filenames in os.walk(data_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTENSIONS:
                continue
            full = os.path.join(root, fname)
            rel  = os.path.relpath(full, data_dir)
            parts = rel.split(os.sep)
            files.append({
                "full_path":     full,
                "file_name":     fname,
                "subject":       parts[0] if len(parts) > 1 else "Unknown",
                "module":        parts[1] if len(parts) > 2 else "General",
                "relative_path": rel,
            })

    logger.info("Found %d supported files in %s", len(files), data_dir)
    return files


get_pdf_files_recursive = get_supported_files  # backward compat


def get_organization_structure(data_dir: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
    if data_dir is None:
        data_dir = str(get_config().data_dir)

    files = get_supported_files(data_dir)
    structure: Dict[str, Dict[str, List[str]]] = {}
    for f in files:
        structure.setdefault(f["subject"], {}).setdefault(f["module"], []).append(f["file_name"])
    return structure