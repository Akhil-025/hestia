"""
core/transformer_classifier.py

A fine-tuned distilbert-sized intent classifier (backlog #25).

Training happens in ``scripts/finetune_classifier.py``, never inside Hestia: a
fine-tune takes minutes to an hour, wants a GPU (an RTX 4050 is plenty; the
script also runs on CPU, slowly) and downloads a base model. Hestia only LOADS
what the script saved:

    <model_dir>/labels.json     classes, thresholds, base model, scores
    <model_dir>/config.json, model.safetensors, tokenizer files   (transformers)

``torch`` and ``transformers`` are imported only when a model is trained or
loaded, so nothing else in Hestia needs them. Like the other backends it
classifies the INTENT only and declines via ``intent_classifier.decide``.

Honest expectation: with ~800 examples across ~230 intents, many with a single
example, a fine-tuned transformer is not guaranteed to beat the TF-IDF model.
That is what the promotion gate in the script is for: it keeps the fine-tune
out unless it measurably wins on the held-out golden prompts.
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from core.intent_classifier import (
    DEFAULT_MIN_MARGIN, DEFAULT_MIN_PROB, Example, Prediction, TrainStats, decide,
)

logger = logging.getLogger(__name__)

LABELS_FILE = "labels.json"
LABELS_VERSION = 1
DEFAULT_BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 48                      # queries are short; this keeps batches small

Scorer = Callable[[Sequence[str]], "object"]    # texts -> (n, n_classes) logits


class TransformerIntentClassifier:
    """Wraps a ``scorer`` (texts -> logits) and the class list that goes with it."""

    def __init__(self, scorer: Scorer, classes: Sequence[str],
                 thresholds: Optional[tuple[float, float]] = None,
                 stats: Optional[TrainStats] = None) -> None:
        self._scorer = scorer
        self._classes = list(classes)
        self.thresholds = thresholds
        self.stats = stats or TrainStats(backend="transformer", classes=len(self._classes))

    @property
    def trained(self) -> bool:
        return self._scorer is not None and bool(self._classes)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def default_thresholds(self) -> tuple[float, float]:
        return self.thresholds or (DEFAULT_MIN_PROB, DEFAULT_MIN_MARGIN)

    def predict(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        if not self.trained or not (text or "").strip():
            return []
        import numpy as np
        logits = np.asarray(self._scorer([text]), dtype=np.float64)[0]
        if logits.shape[0] != len(self._classes):
            raise ValueError("the model's output size does not match its class list")
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        order = np.argsort(-p)[: max(1, k)]
        return [(self._classes[i], float(p[i])) for i in order]

    def classify(self, text: str, min_prob: Optional[float] = None,
                 min_margin: Optional[float] = None) -> Optional[Prediction]:
        d_prob, d_margin = self.default_thresholds
        return decide(self.predict(text, k=3),
                      d_prob if min_prob is None else min_prob,
                      d_margin if min_margin is None else min_margin)

    @classmethod
    def load(cls, model_dir: "str | Path", device: str = "cpu",
             scorer: Optional[Scorer] = None, **_ignored) -> "TransformerIntentClassifier":
        """Raises on a missing/odd directory or when torch/transformers are not
        installed (the service catches and reports)."""
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / LABELS_FILE).read_text(encoding="utf-8"))
        if meta.get("version") != LABELS_VERSION:
            raise ValueError(f"unsupported fine-tuned model version {meta.get('version')!r}")
        th = meta.get("thresholds")
        st = meta.get("stats") or {}
        stats = TrainStats(**{k: st[k] for k in TrainStats.__dataclass_fields__ if k in st})
        stats.backend = "transformer"
        return cls(scorer or load_torch_scorer(model_dir, device),
                   meta["classes"], (float(th[0]), float(th[1])) if th else None, stats)


def write_labels(model_dir: "str | Path", classes: Sequence[str],
                 thresholds: Optional[tuple[float, float]], base_model: str,
                 stats: TrainStats, extra: Optional[dict] = None) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {"version": LABELS_VERSION, "classes": list(classes), "base_model": base_model,
               "thresholds": list(thresholds) if thresholds else None,
               "stats": stats.__dict__, **(extra or {})}
    tmp = model_dir / (LABELS_FILE + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(model_dir / LABELS_FILE)


# ---------------------------------------------------------------------------
# torch-dependent parts (not exercised by the unit tests: no torch in CI)
# ---------------------------------------------------------------------------

def resolve_device(requested: str = "auto") -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was asked for but torch can't see a GPU "
                           "(is this the CPU-only torch build?)")
    return requested


def load_torch_scorer(model_dir: "str | Path", device: str = "cpu", max_length: int = MAX_LENGTH) -> Scorer:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device).eval()

    def scorer(texts: Sequence[str]):
        with torch.no_grad():
            enc = tok(list(texts), padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
            return model(**enc).logits.float().cpu().numpy()
    return scorer


def train_transformer(
    train: Sequence[Example],
    val: Sequence[Example],
    out_dir: "str | Path",
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 5e-5,
    head_lr: float = 1e-3,
    device: str = "auto",
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict:
    """Fine-tune *base_model* on *train*, keep the epoch that did best on *val*,
    and save model + tokenizer into *out_dir* (the caller writes ``labels.json``
    after calibrating thresholds on *val*).

    The classification head starts random with ~230 outputs, so it gets its own,
    larger learning rate. Per-example weights scale the loss, so hand labels
    count for more and augmented variants for less. Plain PyTorch, no Trainer.

    Returns ``{"classes", "best_val_accuracy", "epochs_run", "device", "seconds"}``.
    NOT covered by automated tests (needs torch and a model download); the first
    real run is the test, which is why the script gates what it keeps.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    t0 = time.perf_counter()
    device = resolve_device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    classes = sorted({e.intent for e in train})
    index = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=len(classes))
    model.to(device)

    head = [p for n, p in model.named_parameters() if "classifier" in n]
    body = [p for n, p in model.named_parameters() if "classifier" not in n]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": head_lr}], weight_decay=0.01)
    steps = max(1, math.ceil(len(train) / batch_size) * epochs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def batches(items, shuffle):
        order = list(range(len(items)))
        if shuffle:
            random.shuffle(order)
        for s in range(0, len(order), batch_size):
            yield [items[i] for i in order[s:s + batch_size]]

    def encode(batch):
        enc = tok([e.text for e in batch], padding=True, truncation=True,
                  max_length=MAX_LENGTH, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    def val_accuracy():
        scored = [e for e in val if e.intent in index]
        if not scored:
            return 0.0
        model.eval()
        hit = 0
        with torch.no_grad():
            for b in batches(scored, False):
                pred = model(**encode(b)).logits.argmax(dim=-1).cpu().tolist()
                hit += sum(index[e.intent] == p for e, p in zip(b, pred))
        return hit / len(scored)

    best_acc, best_state, ran = -1.0, None, 0
    n_batches = max(1, math.ceil(len(train) / batch_size))
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for b in batches(list(train), True):
            y = torch.tensor([index[e.intent] for e in b], device=device)
            w = torch.tensor([float(e.weight) for e in b], device=device)
            loss = (loss_fn(model(**encode(b)).logits, y) * w).sum() / w.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += float(loss)
        acc = val_accuracy()
        ran = epoch
        log(f"  epoch {epoch}/{epochs}  loss {total / n_batches:.3f}  val accuracy {acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return {"classes": classes, "best_val_accuracy": round(best_acc, 4), "epochs_run": ran,
            "device": device, "seconds": round(time.perf_counter() - t0, 1)}
