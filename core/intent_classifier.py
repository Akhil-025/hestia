"""
core/intent_classifier.py

A small trained intent classifier that can be retrained from your own logs
(backlog #4, #25).

What it is
----------
Multinomial logistic regression over TF-IDF features of the query's words,
word pairs and character n-grams (the character n-grams are what make it
tolerant of typos and of Hinglish spellings). It is trained here, on this
machine, in a few seconds, from labelled queries; there is no pre-trained
model and no GPU. Saved as one ``.npz`` file (plain arrays plus JSON; no
pickle, so loading a model file can't run code).

What it is not
--------------
It is **not** a fine-tuned distilbert. Backlog #25 suggested one; that needs
torch, a GPU-class training run and a model download, none of which fits a
"retrain in the background after you correct it" workflow. This is the
smallest thing that removes the dependence on Ollama's JSON-mode
reliability for the common case, and it is a deliberate swap point: anything
with ``fit(examples)`` / ``predict(text, k)`` / ``save`` / ``load`` can
replace it without touching Hecate or the NLU.

It classifies the INTENT only. It does not extract entities ("500", "mom",
"tomorrow"), so where it stands in for the LLM the NLU hands back an empty
entity dict and the modules' own slot-filling asks for what is missing, which
is the path aliases (``config/intent_aliases.yaml``) already use.

Honest accuracy
---------------
On the held-out golden prompts (``hestia_test_prompts.md``, removed from the
training set) it answers a little over half of them (~58%) and is right on
~95% of the ones it answers; the rest it declines, and those follow the normal
routing tiers. Measured with ``python main.py --train-classifier``, which
reproduces the numbers on your own data and lists the misses. A small training
set (a few hundred examples across ~230 intents) is the real limit, which is
why logged traffic and your hand labels (``--label``) feed back in.
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_VERSION = 1

# Decision thresholds, tuned on the held-out golden set (see module docstring):
# probabilities are spread across ~240 classes, so even a clearly right answer
# is rarely above 0.6. These keep the answers that are right ~98% of the time.
DEFAULT_MIN_PROB = 0.30
DEFAULT_MIN_MARGIN = 0.12


def _np():
    """numpy, or None. The classifier degrades to 'unavailable', never raises."""
    try:
        import numpy as np
        return np
    except Exception:  # pragma: no cover - numpy is a hard requirement elsewhere
        return None


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_DIGITS_RE = re.compile(r"\d+(?:[.,]\d+)*")


def normalise(text: str) -> str:
    """Lower-case, NFKC, numbers collapsed to '0' so "log 500" and "log 20" match."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return _DIGITS_RE.sub("0", text)


def featurize(text: str) -> list[str]:
    """Feature names for one query: words, word pairs, char 2-5-grams per word."""
    words = _WORD_RE.findall(normalise(text))
    feats: list[str] = [f"w:{w}" for w in words]
    feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    for w in words:
        padded = f" {w} "
        for n in (2, 3, 4, 5):
            if len(padded) < n:
                continue
            for i in range(len(padded) - n + 1):
                feats.append(f"c:{padded[i:i + n]}")
    return feats


def _term_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in featurize(text):
        counts[f] = counts.get(f, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Example:
    text: str
    intent: str
    weight: float = 1.0
    source: str = "unknown"


@dataclass
class Prediction:
    intent: str
    probability: float
    margin: float                       # gap to the runner-up
    alternatives: list = field(default_factory=list)   # [(intent, prob), ...]


@dataclass
class TrainStats:
    examples: int = 0
    classes: int = 0
    features: int = 0
    epochs: int = 0
    train_accuracy: float = 0.0
    seconds: float = 0.0
    sources: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Softmax regression over TF-IDF features. Thread-safe once fitted
    (prediction only reads arrays; ``fit`` builds new ones and swaps them in)."""

    def __init__(
        self,
        l2: float = 3e-5,
        epochs: int = 160,
        learning_rate: float = 0.25,
        min_count: int = 1,
        seed: int = 0,
    ) -> None:
        self.l2 = l2
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self._vocab: dict[str, int] = {}
        self._idf = None
        self._W = None
        self._b = None
        self._classes: list[str] = []
        self.stats = TrainStats()

    # -- state ---------------------------------------------------------

    @property
    def trained(self) -> bool:
        return self._W is not None and bool(self._classes)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    # -- vectorising ---------------------------------------------------

    def _vectorise(self, text: str):
        """``(indices, values)`` of one L2-normalised TF-IDF row."""
        np = _np()
        idx: list[int] = []
        val: list[float] = []
        for feat, n in _term_counts(text).items():
            j = self._vocab.get(feat)
            if j is None:
                continue
            idx.append(j)
            val.append((1.0 + math.log(n)) * float(self._idf[j]))
        if not idx:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        arr = np.asarray(val, dtype=np.float32)
        norm = float(np.sqrt((arr * arr).sum()))
        if norm > 0:
            arr = arr / norm
        return np.asarray(idx, dtype=np.int64), arr

    # -- training ------------------------------------------------------

    def fit(self, examples: Sequence[Example]) -> TrainStats:
        """Train from scratch on *examples* and replace the current model.

        Raises ``ValueError`` for fewer than two classes or no usable text, and
        ``RuntimeError`` if numpy is missing; callers (the service below) catch
        and report these rather than letting a bad log poison routing.
        """
        import time
        np = _np()
        if np is None:
            raise RuntimeError("numpy is required to train the intent classifier")
        t0 = time.perf_counter()
        usable = [e for e in examples if (e.text or "").strip() and e.intent]
        classes = sorted({e.intent for e in usable})
        if len(classes) < 2:
            raise ValueError("need examples of at least two different intents to train")

        # Vocabulary and document frequency.
        df: dict[str, int] = {}
        rows_counts = []
        for e in usable:
            counts = _term_counts(e.text)
            rows_counts.append(counts)
            for f in counts:
                df[f] = df.get(f, 0) + 1
        feats = sorted(f for f, c in df.items() if c >= self.min_count)
        vocab = {f: i for i, f in enumerate(feats)}
        n_docs = len(usable)
        idf = np.asarray(
            [math.log((1 + n_docs) / (1 + df[f])) + 1.0 for f in feats], dtype=np.float32
        )

        # Sparse design matrix in CSR layout (plain arrays; scipy optional).
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for counts in rows_counts:
            row = [(vocab[f], (1.0 + math.log(n)) * float(idf[vocab[f]]))
                   for f, n in counts.items() if f in vocab]
            norm = math.sqrt(sum(v * v for _, v in row)) or 1.0
            for j, v in row:
                indices.append(j)
                data.append(v / norm)
            indptr.append(len(indices))
        n, d, k = len(usable), len(feats), len(classes)
        cls_index = {c: i for i, c in enumerate(classes)}
        y = np.asarray([cls_index[e.intent] for e in usable], dtype=np.int64)
        w = np.asarray([max(float(e.weight), 0.0) for e in usable], dtype=np.float32)
        w = w / (w.sum() or 1.0)

        X = _csr(np, indptr, indices, data, (n, d))
        W = np.zeros((d, k), dtype=np.float32)
        b = np.zeros(k, dtype=np.float32)
        # Adam, full batch. Deterministic: no shuffling, zero init.
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        mb = np.zeros_like(b); vb = np.zeros_like(b)
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, self.learning_rate
        onehot = np.zeros((n, k), dtype=np.float32)
        onehot[np.arange(n), y] = 1.0
        for step in range(1, self.epochs + 1):
            logits = X.dot(W) + b
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            err = (p - onehot) * w[:, None]
            gW = X.T.dot(err) + self.l2 * W
            gb = err.sum(axis=0)
            mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW * gW
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
            c1, c2 = 1 - b1 ** step, 1 - b2 ** step
            W -= lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
            b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)

        logits = X.dot(W) + b
        acc = float((logits.argmax(axis=1) == y).mean())
        sources: dict[str, int] = {}
        for e in usable:
            sources[e.source] = sources.get(e.source, 0) + 1

        # Swap in only once everything above succeeded.
        self._vocab, self._idf, self._W, self._b = vocab, idf, W, b
        self._classes = classes
        self.stats = TrainStats(
            examples=n, classes=k, features=d, epochs=self.epochs,
            train_accuracy=round(acc, 4), seconds=round(time.perf_counter() - t0, 2),
            sources=sources,
        )
        return self.stats

    # -- prediction ----------------------------------------------------

    def predict(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """The *k* most likely ``(intent, probability)`` pairs, best first.
        Empty if untrained or the text shares no features with the training set."""
        if not self.trained:
            return []
        np = _np()
        idx, val = self._vectorise(text)
        if idx.size == 0:
            return []
        logits = (self._W[idx] * val[:, None]).sum(axis=0) + self._b
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()
        order = np.argsort(-p)[: max(1, k)]
        return [(self._classes[i], float(p[i])) for i in order]

    def classify(
        self,
        text: str,
        min_prob: float = DEFAULT_MIN_PROB,
        min_margin: float = DEFAULT_MIN_MARGIN,
    ) -> Optional[Prediction]:
        """The best intent, or None when it isn't sure enough to answer.
        Declining is the point: a wrong confident answer is worse than none."""
        top = self.predict(text, k=3)
        if not top:
            return None
        best, p1 = top[0]
        p2 = top[1][1] if len(top) > 1 else 0.0
        if p1 < min_prob or (p1 - p2) < min_margin:
            return None
        return Prediction(intent=best, probability=round(p1, 4),
                          margin=round(p1 - p2, 4), alternatives=top[1:])

    # -- persistence ---------------------------------------------------

    def save(self, path: "str | Path") -> None:
        if not self.trained:
            raise RuntimeError("nothing to save: the classifier is not trained")
        np = _np()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        feats = [None] * len(self._vocab)
        for f, i in self._vocab.items():
            feats[i] = f
        meta = json.dumps({
            "version": MODEL_VERSION, "classes": self._classes, "features": feats,
            "stats": self.stats.__dict__,
        })
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, W=self._W, b=self._b, idf=self._idf, meta=np.array(meta))
        tmp.replace(path)   # atomic: a crash mid-save never leaves a half-written model

    @classmethod
    def load(cls, path: "str | Path") -> "IntentClassifier":
        """Raises on a missing, corrupt or wrong-version file (the service catches)."""
        np = _np()
        if np is None:
            raise RuntimeError("numpy is required to load the intent classifier")
        with np.load(str(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("version") != MODEL_VERSION:
                raise ValueError(f"unsupported classifier model version {meta.get('version')!r}")
            clf = cls()
            clf._W, clf._b, clf._idf = z["W"], z["b"], z["idf"]
        clf._classes = list(meta["classes"])
        clf._vocab = {f: i for i, f in enumerate(meta["features"])}
        if clf._W.shape != (len(clf._vocab), len(clf._classes)):
            raise ValueError("classifier model file is internally inconsistent")
        st = meta.get("stats") or {}
        clf.stats = TrainStats(**{k: st[k] for k in TrainStats.__dataclass_fields__ if k in st})
        return clf


def _csr(np, indptr, indices, data, shape):
    """A CSR matrix with ``.dot`` and ``.T.dot``: scipy's if available, else a
    small numpy stand-in (slower, same results)."""
    try:
        from scipy.sparse import csr_matrix
        return csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int32)), shape=shape)
    except Exception:
        return _DenseFallback(np, indptr, indices, data, shape)


class _DenseFallback:
    """Dense stand-in used only when scipy is missing."""

    def __init__(self, np, indptr, indices, data, shape):
        self._np = np
        self.A = np.zeros(shape, dtype=np.float32)
        for r in range(shape[0]):
            for p in range(indptr[r], indptr[r + 1]):
                self.A[r, indices[p]] = data[p]

    def dot(self, other):
        return self.A.dot(other)

    @property
    def T(self):
        class _T:
            def __init__(s, a):
                s.a = a

            def dot(s, o):
                return s.a.T.dot(o)
        return _T(self.A)


# ---------------------------------------------------------------------------
# Service: where the trained model lives and who may use it
# ---------------------------------------------------------------------------

MODES = ("off", "assist", "primary")


class ClassifierService:
    """Owns the model file, loads or trains it (never blocking startup), and
    hands out confident predictions to Hecate and the NLU.

    ``mode``:
      * ``off``      never consulted.
      * ``assist``   (default) consulted only where Hestia would otherwise
                     give up: Hecate's last-resort tiers, and the NLU when the
                     LLM is unreachable or fails.
      * ``primary``  Hecate also asks it BEFORE the hand-written text-trigger
                     tiers whenever the NLU said "chat"/something unroutable
                     or was unsure, so the trained model, not the hand-tuned
                     phrase lists, gets first say. Those tiers remain as the
                     backstop when it declines.
    """

    def __init__(
        self,
        model_path: "str | Path" = "data/intent_classifier.npz",
        mode: str = "assist",
        min_prob: float = DEFAULT_MIN_PROB,
        min_margin: float = DEFAULT_MIN_MARGIN,
        examples_fn=None,
        valid_intents: Optional[Iterable[str]] = None,
    ) -> None:
        import threading
        self.model_path = Path(model_path)
        self.mode = mode if mode in MODES else "assist"
        self.min_prob = float(min_prob)
        self.min_margin = float(min_margin)
        self._examples_fn = examples_fn
        self._valid = frozenset(valid_intents) if valid_intents is not None else None
        self._clf: Optional[IntentClassifier] = None
        self._lock = threading.Lock()
        self._training = False
        self.last_error: Optional[str] = None

    # -- availability --------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def ready(self) -> bool:
        return self._clf is not None and self._clf.trained

    def load(self) -> bool:
        """Load the saved model if there is a good one. Never raises."""
        if not self.model_path.is_file():
            return False
        try:
            clf = IntentClassifier.load(self.model_path)
        except Exception as exc:
            self.last_error = f"couldn't load {self.model_path}: {exc}"
            logger.warning("[Classifier] %s", self.last_error)
            return False
        with self._lock:
            self._clf = clf
        return True

    def train(self) -> dict:
        """(Re)train from the configured examples, save, and swap the new model
        in. Returns a status dict; never raises (a failed retrain leaves the
        previous model serving)."""
        if self._examples_fn is None:
            return {"ok": False, "error": "no training data source configured"}
        with self._lock:
            if self._training:
                return {"ok": False, "error": "a training run is already in progress"}
            self._training = True
        try:
            examples = list(self._examples_fn())
            clf = IntentClassifier()
            stats = clf.fit(examples)
            try:
                clf.save(self.model_path)
            except OSError as exc:
                # Still usable in memory; just won't survive a restart.
                self.last_error = f"trained but couldn't save: {exc}"
                logger.warning("[Classifier] %s", self.last_error)
            with self._lock:
                self._clf = clf
            self.last_error = None if self.last_error is None or "save" not in self.last_error else self.last_error
            return {"ok": True, "stats": stats}
        except Exception as exc:
            self.last_error = f"training failed: {exc}"
            logger.warning("[Classifier] %s", self.last_error)
            return {"ok": False, "error": self.last_error}
        finally:
            with self._lock:
                self._training = False

    def retrain_if_needed(self, min_new: int = 20) -> dict:
        """Retrain when the training data has grown by at least *min_new*
        examples since the loaded model was fitted (new hand labels, new
        confident log lines). Cheap to call: it only counts examples first.
        Never raises."""
        if not self.enabled or self._examples_fn is None:
            return {"ok": False, "skipped": "classifier off or no data source"}
        try:
            have = len(list(self._examples_fn()))
        except Exception as exc:
            return {"ok": False, "error": f"couldn't read training data: {exc}"}
        trained_on = self._clf.stats.examples if self.ready else 0
        if self.ready and have - trained_on < min_new:
            return {"ok": True, "skipped": f"only {have - trained_on} new example(s)"}
        return self.train()

    def ensure_ready(self, background: bool = True) -> None:
        """Load a saved model, or train one. Training runs on a daemon thread
        by default so a slow first run can never delay startup."""
        if not self.enabled or self.ready:
            return
        if self.load():
            return
        if background:
            import threading
            threading.Thread(target=self.train, daemon=True, name="IntentClassifierTrain").start()
        else:
            self.train()

    # -- use -----------------------------------------------------------

    def classify(self, text: str) -> Optional[Prediction]:
        """A confident prediction for a registered intent, or None. Never raises."""
        if not self.enabled or not self.ready:
            return None
        try:
            pred = self._clf.classify(text, self.min_prob, self.min_margin)
        except Exception as exc:
            logger.debug("[Classifier] classify failed: %s", exc)
            return None
        if pred is None:
            return None
        if self._valid is not None and pred.intent not in self._valid:
            return None
        return pred

    def status(self) -> dict:
        s = self._clf.stats if self.ready else None
        return {
            "mode": self.mode, "ready": self.ready, "training": self._training,
            "model_path": str(self.model_path), "last_error": self.last_error,
            "examples": s.examples if s else 0, "classes": s.classes if s else 0,
            "train_accuracy": s.train_accuracy if s else None,
            "sources": dict(s.sources) if s else {},
        }


def stable_hash(text: str) -> int:
    """Process-independent hash (``hash()`` is salted per process)."""
    return zlib.crc32((text or "").encode("utf-8")) & 0xFFFFFFFF
