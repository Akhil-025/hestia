"""
modules/pluto/receipts.py

Log an expense from a photo of a receipt (backlog #141).

Pipeline: image file -> OCR text -> pick out merchant, total and date ->
save through the same ``expenses`` table ``log_expense`` writes to.

On "pairs with Iris's OCR pipeline": Iris has no OCR step in this codebase
(it captions and embeds images), so this module runs pytesseract itself, the
same library Athena uses for scanned PDFs. ``pytesseract`` and Pillow are in
requirements.txt; the Tesseract program must be installed separately. When
either is missing the reply says so rather than failing.

OCR misreads digits, so a wrong total is the main risk. The rules:

* A total is trusted only when it sits on a line labelled like a total
  ("grand total", "amount payable", "total", ...). Subtotal, tax, discount,
  change and tender lines are never taken for it.
* With no labelled total the reply shows what was read and asks for the
  amount instead of guessing from "the biggest number on the page".
* The reply always states what was saved (amount, merchant, date) so a
  misread is visible at once.
* Same amount and merchant already logged for that day is reported as a
  likely duplicate and not saved, unless the caller confirms.

The receipt image itself is never copied or stored by this module.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from .logging_config import get_logger
from .planning import MAX_AMOUNT

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_OCR_CHARS = 20_000

# Earlier = stronger. The last line matching the strongest present label wins.
# "amount paid" and "balance due" come last: they can be the cash handed over or
# what is left after a part payment, not the bill.
_TOTAL_LABELS = [
    r"grand\s*total", r"net\s*(?:amount\s*)?payable", r"total\s*(?:amount\s*)?payable",
    r"amount\s*(?:payable|due)", r"net\s*amount", r"total\s*amount", r"total",
    r"balance\s*due", r"amount\s*paid",
]
_SUBTOTAL = re.compile(r"sub\s*-?\s*total|items?\s*total", re.I)
# For a bare "total" line only: these words mean it is a tax/discount/etc. total, not the bill.
_NOT_TOTAL = re.compile(
    r"sub\s*-?\s*total|tax|gst|cgst|sgst|igst|vat|cess|discount|saving|round\s*off|"
    r"change|tender|cash\s+given|points|tip\b|qty|quantity|items?\s*total",
    re.I,
)
_AMOUNT = re.compile(r"(?<![\w.,])(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)(?![\w])", re.I)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_SKIP_MERCHANT = re.compile(
    r"\b(?:invoice|receipt|tax|gstin|bill|cash\s*memo|welcome|thank|thanks|order|table|date|time|"
    r"phone|ph|tel|mob|mobile)\b|www\.|@|^\W*$|^\d", re.I)


class ReceiptError(Exception):
    """Something the user can act on (missing file, OCR not installed, ...)."""


@dataclass
class ParsedReceipt:
    merchant: Optional[str] = None
    total: Optional[float] = None
    receipt_date: Optional[date] = None
    total_labelled: bool = False        # True only when the total came from a total-type line
    notes: list[str] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return self.total is not None and self.total_labelled


def _to_amount(text: str) -> Optional[float]:
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) and 0 < value <= MAX_AMOUNT else None


def _amounts_in(line: str) -> list[float]:
    out = []
    for m in _AMOUNT.finditer(line):
        v = _to_amount(m.group(1))
        if v is not None:
            out.append(v)
    return out


def find_total(lines: list[str]) -> tuple[Optional[float], bool]:
    """(amount, came_from_a_total_line). Falls back to the largest amount, unlabelled."""
    for label in _TOTAL_LABELS:
        pattern = re.compile(label, re.I)
        found: Optional[float] = None
        for i, line in enumerate(lines):
            if not pattern.search(line) or _SUBTOTAL.search(line) \
                    or (label == r"total" and _NOT_TOTAL.search(line)):
                continue
            amounts = _amounts_in(line)
            if not amounts and i + 1 < len(lines) and not _NOT_TOTAL.search(lines[i + 1]):
                amounts = _amounts_in(lines[i + 1])          # label and figure on separate lines
            if amounts:
                found = amounts[-1]                          # figure is usually last on the line
        if found is not None:
            return found, True
    every = [a for line in lines for a in _amounts_in(line)]
    return (max(every), False) if every else (None, False)


def find_date(text: str, today: Optional[date] = None) -> Optional[date]:
    """First plausible date: day-first numeric (Indian receipts), ISO, or '12 Mar 2026'.

    Dates in the future, or more than two years back, are ignored as OCR noise.
    """
    today = today or date.today()

    def ok(y: int, m: int, d: int) -> Optional[date]:
        if y < 100:
            y += 2000
        try:
            found = date(y, m, d)
        except ValueError:
            return None
        return found if 0 <= (today - found).days <= 731 else None

    for m in re.finditer(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", text):
        d = ok(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            return d
    for m in re.finditer(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})(?!\d)", text):
        d = ok(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if d:
            return d
    for m in re.finditer(r"(?<!\d)(\d{1,2})[\s\-]*([A-Za-z]{3})[a-z]*[\s,\-]*(\d{2}|\d{4})(?!\d)", text):
        mon = _MONTHS.get(m.group(2).lower())
        if mon:
            d = ok(int(m.group(3)), mon, int(m.group(1)))
            if d:
                return d
    return None


def find_merchant(lines: list[str]) -> Optional[str]:
    """First of the top few lines that looks like a shop name."""
    for line in lines[:6]:
        clean = re.sub(r"[^\w&'.\- ]", " ", line).strip()
        clean = re.sub(r"\s+", " ", clean)
        if len(clean) >= 3 and sum(c.isalpha() for c in clean) >= 3 and not _SKIP_MERCHANT.search(clean):
            return clean[:60].title() if clean.isupper() else clean[:60]
    return None


def parse_receipt_text(text: str, today: Optional[date] = None) -> ParsedReceipt:
    text = (text or "")[:MAX_OCR_CHARS]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    parsed = ParsedReceipt()
    if not lines:
        parsed.notes.append("The photo gave no readable text.")
        return parsed
    parsed.total, parsed.total_labelled = find_total(lines)
    parsed.receipt_date = find_date(text, today)
    parsed.merchant = find_merchant(lines)
    if parsed.total is None:
        parsed.notes.append("I couldn't find an amount.")
    elif not parsed.total_labelled:
        parsed.notes.append("No line was labelled as the total, so this amount is only the largest figure I saw.")
    return parsed


def tesseract_ocr(path: Path) -> str:
    """Default OCR: Pillow + pytesseract. Raises ReceiptError with an install hint if absent."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise ReceiptError("Reading receipts needs pytesseract and Pillow (pip install pytesseract pillow) "
                           "plus the Tesseract program.") from e
    try:
        with Image.open(path) as img:
            img.load()
            return pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError as e:
        raise ReceiptError("Tesseract isn't installed on this machine, so I can't read the photo.") from e
    except Exception as e:
        raise ReceiptError("I couldn't open that file as an image.") from e


class ReceiptIngestor:
    def __init__(self, db: Any = None, currency: str = "\u20b9",
                 ocr_fn: Optional[Callable[[Path], str]] = None,
                 category_fn: Optional[Callable[[str, float], str]] = None,
                 sanitize_fn: Optional[Callable[[str], str]] = None):
        self.db = db
        self.currency = currency
        self.ocr_fn = ocr_fn or tesseract_ocr
        self.category_fn = category_fn
        self.sanitize_fn = sanitize_fn or (lambda s: re.sub(r"[\x00-\x1f\x7f]", " ", s).strip())

    @staticmethod
    def check_image(path_text: str) -> Path:
        """Validate a user-supplied path. Raises ReceiptError."""
        if not path_text or not str(path_text).strip():
            raise ReceiptError("Which receipt photo? Give me the file path, or upload it in the web UI.")
        path = Path(str(path_text).strip().strip("\"'")).expanduser()
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ReceiptError("That doesn't look like an image file (jpg, png, webp, bmp or tiff).")
        try:
            if not path.is_file():
                raise ReceiptError(f"I can't find a file at {path}.")
            size = path.stat().st_size
        except OSError:
            raise ReceiptError(f"I can't open {path}.")
        if size == 0 or size > MAX_IMAGE_BYTES:
            raise ReceiptError("That image is empty or larger than 10 MB.")
        return path

    def ingest(self, entities: dict, today: Optional[date] = None) -> dict:
        if self.db is None:
            return _err("I can't reach the finance database right now.")
        try:
            path = self.check_image(entities.get("image_path") or entities.get("path") or entities.get("file") or "")
            text = self.ocr_fn(path)
        except ReceiptError as e:
            return _err(str(e))
        except Exception:
            logger.exception("log_receipt: OCR failed")
            return _err("Something went wrong while reading that photo.")
        return self.ingest_text(text, entities, today)

    def ingest_text(self, text: str, entities: dict, today: Optional[date] = None) -> dict:
        """The part after OCR, separate so it can be tested without an image."""
        from .planning import parse_amount

        parsed = parse_receipt_text(text, today)
        confirm = str(entities.get("confirm", "")).lower() in ("1", "true", "yes", "y")

        override = entities.get("amount")
        if override not in (None, ""):
            amount = parse_amount(override)
            if amount is None:
                return _err(f"I couldn't read {override!r} as an amount.")
            parsed.total, parsed.total_labelled = amount, True
            parsed.notes = [n for n in parsed.notes if "amount" not in n]

        merchant = self.sanitize_fn(str(entities.get("merchant") or parsed.merchant or "receipt"))[:60] or "receipt"
        read = self._read_back(parsed, merchant)

        if not parsed.confident:
            if parsed.total is None:
                return {"response": f"I couldn't read a total from that receipt. {read}"
                                    "Tell me the amount, e.g. 'log receipt for 450'.",
                        "data": {"merchant": merchant}, "confidence": 0.3}
            return {"response": f"{read}I'm not sure {self._fmt(parsed.total)} is the total, so I haven't saved "
                                "anything. If it is, say 'log receipt, amount "
                                f"{parsed.total:g}'; otherwise give me the right amount.",
                    "data": {"merchant": merchant, "candidate_amount": parsed.total, "saved": False},
                    "confidence": 0.4}

        when = parsed.receipt_date
        description = f"{merchant} (receipt)"
        logged_at = f"{when.isoformat()} 12:00:00" if when else None

        if not confirm and self._duplicate(parsed.total, description, when, today):
            return {"response": f"{read}That looks like it's already logged (same amount and shop that day), "
                                "so I haven't saved it again. Say 'log receipt, confirm' to save it anyway.",
                    "data": {"merchant": merchant, "amount": parsed.total, "saved": False, "duplicate": True},
                    "confidence": 0.6}

        category = "other"
        if self.category_fn:
            try:
                category = self.category_fn(description, parsed.total) or "other"
            except Exception:
                logger.exception("log_receipt: category inference failed")
        try:
            if logged_at:
                self.db.log_expense(parsed.total, description, category, logged_at=logged_at)
            else:
                self.db.log_expense(parsed.total, description, category)
        except Exception:
            logger.exception("log_receipt: DB write failed")
            return _err("I read the receipt but couldn't save the expense. Please try again.")

        dated = f" on {when.isoformat()}" if when else ""
        return {"response": f"Logged {self._fmt(parsed.total)} for {description} under {category}{dated}. "
                            "Check the amount against the receipt; photos can be misread.",
                "data": {"amount": parsed.total, "description": description, "category": category,
                         "date": when.isoformat() if when else None, "saved": True},
                "confidence": 0.85}

    # ---- helpers --------------------------------------------------------

    def _read_back(self, p: ParsedReceipt, merchant: str) -> str:
        bits = [f"shop: {merchant}"]
        if p.total is not None:
            bits.append(f"amount: {self._fmt(p.total)}")
        if p.receipt_date:
            bits.append(f"date: {p.receipt_date.isoformat()}")
        return "I read " + ", ".join(bits) + ". "

    def _duplicate(self, amount: float, description: str, when: Optional[date], today: Optional[date]) -> bool:
        day = when or today or date.today()
        try:
            # end is exclusive in PlutoDB.get_expenses_between
            rows = self.db.get_expenses_between(day.isoformat(), (day + timedelta(days=1)).isoformat())
        except Exception:
            return False
        return any(abs(float(r.get("amount") or 0) - amount) < 0.005
                   and str(r.get("description", "")).lower() == description.lower() for r in rows)

    def _fmt(self, amount: float) -> str:
        return f"{self.currency}{amount:,.2f}"


def _err(response: str) -> dict:
    return {"response": response, "data": {}, "confidence": 0.0}
