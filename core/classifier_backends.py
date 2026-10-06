"""
core/classifier_backends.py

Alternative intent classifiers behind the same interface as the TF-IDF model in
``core/intent_classifier.py`` (backlog #25), plus the machinery to calibrate
their decision thresholds without touching the held-out golden prompts.

Backends (``classifier.backend`` in the config):

  tfidf       the original model. Default. No downloads.
  embedding   sentence-embedding nearest neighbours (all-MiniLM-L6-v2, the model
              the rest of Hestia already uses). Understands paraphrase the
              word/character model cannot. Needs ``sentence-transformers``.
  ensemble    the two, probabilities averaged.
  transformer a fine-tuned distilbert-sized model (``core/transformer_classifier.py``,
              produced by ``scripts/finetune_classifier.py``).

All classify the INTENT only (no entity extraction) and decline rather than
guess, through the one shared rule ``intent_classifier.decide``.

Embedding model
---------------
Every training example is embedded once and stored. A query's cosine similarity
to every example becomes weights via a softmax (temperature 0.05, so near
matches dominate), summed per intent: a probability over intents, no training
loop to tune.

Thresholds
----------
The TF-IDF cut-offs (0.30 / 0.12) mean something different on this probability
scale, so ``fit`` picks its own: k-fold cross-validation over the TRAINING
examples (an example is scored by a model that has not seen it, nor any
augmented variant of it), then the loosest cut-offs that still reach
``target_precision`` (0.95). The golden prompts are never used to choose
anything, so scoring on them stays an honest test.

Limits: cross-validation on this data is optimistic (many alias lines are
near-duplicates), so expect real precision a little under the target; and
intents with a single example cannot be validated at all.
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Sequence

from core.intent_classifier import (
    Example, IntentClassifier, Prediction, TrainStats, _np, decide, normalise,
)

logger = logging.getLogger(__name__)

BACKENDS = ("tfidf", "embedding", "ensemble", "transformer")
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_VERSION = 1
ENSEMBLE_VERSION = 1
UNCALIBRATED_THRESHOLDS = (0.50, 0.20)      # used only if calibration could not run
AUGMENT_SOURCE = "augment"

Embedder = Callable[[Sequence[str]], "object"]   # texts -> (n, d) array


class SentenceTransformerEmbedder:
    """``sentence-transformers`` behind the ``Embedder`` call signature.
    Loads eagerly so a missing library fails at load time, not mid-query."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def __call__(self, texts: Sequence[str]):
        return self._model.encode(list(texts), normalize_embeddings=True, batch_size=64,
                                  show_progress_bar=False, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Cross-validation and thresholds
# ---------------------------------------------------------------------------

def _key(text: str) -> str:
    return " ".join(normalise(text).split())


def _group(e: Example) -> str:
    return e.parent or _key(e.text)


def cv_predictions(
    examples: Sequence[Example],
    predict_fold: Callable[[list, list], list],
    folds: int = 5,
    seed: int = 0,
) -> list[tuple[str, list]]:
    """``[(true_intent, top_list), ...]`` for held-out examples.

    ``predict_fold(train_idx, test_idx)`` returns one ``[(intent, p), ...]`` per
    test index, using a model built from ``train_idx`` only. An augmented
    variant always lands in the fold of the example it came from, and only real
    (non-augmented) examples are scored, so a variant of a test example is never
    in its training set. Examples of an intent with no other example outside
    their fold are skipped: nothing could have predicted them.
    """
    keys = sorted({_group(e) for e in examples})
    folds = min(folds, len(keys) // 2)
    if folds < 2:
        return []
    order = list(keys)
    random.Random(seed).shuffle(order)
    fold_of = {k: i % folds for i, k in enumerate(order)}
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for i, e in enumerate(examples):
        buckets[fold_of[_group(e)]].append(i)

    out: list[tuple[str, list]] = []
    for f in range(folds):
        train = [i for g in range(folds) if g != f for i in buckets[g]]
        known = {examples[i].intent for i in train}
        test = [i for i in buckets[f]
                if examples[i].source != AUGMENT_SOURCE and examples[i].intent in known]
        if not test or len(known) < 2:
            continue
        for i, top in zip(test, predict_fold(train, test)):
            out.append((examples[i].intent, top))
    return out


_PROB_GRID = [round(0.10 + 0.05 * i, 2) for i in range(17)]            # 0.10 .. 0.90
_MARGIN_GRID = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30, 0.40, 0.50]


def choose_thresholds(
    preds: Sequence[tuple[str, list]],
    target_precision: float = 0.95,
    min_answered: int = 5,
) -> dict:
    """The loosest ``(min_prob, min_margin)`` whose answers are right at least
    *target_precision* of the time, judged on cross-validated predictions. Ties
    go to the stricter pair. If nothing qualifies, the strictest pair is
    returned and ``met_target`` is False (it will then almost never answer)."""
    n = len(preds)
    best = None
    for mp in _PROB_GRID:
        for mm in _MARGIN_GRID:
            answered = right = 0
            for truth, top in preds:
                d = decide(top, mp, mm)
                if d is None:
                    continue
                answered += 1
                right += d.intent == truth
            if answered < min_answered or right / answered < target_precision:
                continue
            cand = (answered, mp + mm, mp, mm, right / answered)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is None:
        return {"min_prob": _PROB_GRID[-1], "min_margin": _MARGIN_GRID[-1], "met_target": False,
                "scored": n, "coverage": 0.0, "precision": 0.0, "target_precision": target_precision}
    answered, _, mp, mm, precision = best
    return {"min_prob": mp, "min_margin": mm, "met_target": True, "scored": n,
            "coverage": round(answered / n, 3) if n else 0.0,
            "precision": round(precision, 3), "target_precision": target_precision}


def _top(probs: dict, k: int) -> list[tuple[str, float]]:
    return sorted(probs.items(), key=lambda kv: -kv[1])[: max(1, k)]


# ---------------------------------------------------------------------------
# Embedding classifier
# ---------------------------------------------------------------------------

class EmbeddingIntentClassifier:
    """Softmax-weighted nearest neighbours over sentence embeddings."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
        temperature: float = 0.05,
        folds: int = 5,
        target_precision: float = 0.95,
        seed: int = 0,
    ) -> None:
        self._embedder = embedder
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.folds = folds
        self.target_precision = target_precision
        self.seed = seed
        self._X = None            # (n, d) float32, rows L2-normalised
        self._y = None            # (n,) int class indices
        self._w = None            # (n,) example weights
        self._classes: list[str] = []
        self.thresholds: Optional[tuple[float, float]] = None
        self.stats = TrainStats(backend="embedding")
        self._cache: "OrderedDict[str, object]" = OrderedDict()

    @property
    def trained(self) -> bool:
        return self._X is not None and bool(self._classes)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def default_thresholds(self) -> tuple[float, float]:
        return self.thresholds or UNCALIBRATED_THRESHOLDS

    @staticmethod
    def usable(examples: Sequence[Example]) -> list[Example]:
        return [e for e in examples if (e.text or "").strip() and e.intent]

    # -- embedding -----------------------------------------------------

    def _embed(self, texts: Sequence[str]):
        np = _np()
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.model_name, self.device)
        arr = np.asarray(self._embedder(list(texts)), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise ValueError("the embedder returned an unexpected shape")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)

    def _embed_query(self, text: str):
        hit = self._cache.get(text)
        if hit is not None:
            self._cache.move_to_end(text)
            return hit
        vec = self._embed([text])[0]
        self._cache[text] = vec
        if len(self._cache) > 256:
            self._cache.popitem(last=False)
        return vec

    # -- scoring -------------------------------------------------------

    def _probs(self, Q, X, y, w, n_classes: int):
        """(len(Q), n_classes) probabilities from query rows Q against stored rows X."""
        np = _np()
        S = (Q @ X.T) / self.temperature
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S) * w[None, :]
        onehot = np.zeros((X.shape[0], n_classes), dtype=np.float32)
        onehot[np.arange(X.shape[0]), y] = 1.0
        P = E @ onehot
        return P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)

    def class_probabilities(self, text: str) -> dict[str, float]:
        """Probability for EVERY class (what the ensemble averages)."""
        if not self.trained or not (text or "").strip():
            return {}
        q = self._embed_query(text)[None, :]
        p = self._probs(q, self._X, self._y, self._w, len(self._classes))[0]
        return {c: float(p[i]) for i, c in enumerate(self._classes)}

    def predict(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        return _top(self.class_probabilities(text), k)

    def classify(self, text: str, min_prob: Optional[float] = None,
                 min_margin: Optional[float] = None) -> Optional[Prediction]:
        d_prob, d_margin = self.default_thresholds
        return decide(self.predict(text, k=3),
                      d_prob if min_prob is None else min_prob,
                      d_margin if min_margin is None else min_margin)

    # -- training ------------------------------------------------------

    def _set(self, usable: Sequence[Example], X) -> None:
        np = _np()
        classes = sorted({e.intent for e in usable})
        index = {c: i for i, c in enumerate(classes)}
        self._classes = classes
        self._X = X.astype(np.float32)
        self._y = np.asarray([index[e.intent] for e in usable], dtype=np.int64)
        self._w = np.asarray([max(float(e.weight), 1e-3) for e in usable], dtype=np.float32)
        self._cache.clear()

    def fit(self, examples: Sequence[Example], calibrate: bool = True) -> TrainStats:
        np = _np()
        if np is None:
            raise RuntimeError("numpy is required to train the embedding classifier")
        t0 = time.perf_counter()
        usable = self.usable(examples)
        if len({e.intent for e in usable}) < 2:
            raise ValueError("need examples of at least two different intents to train")
        X = self._embed([e.text for e in usable])
        self._set(usable, X)

        calibration: dict = {}
        self.thresholds = None
        if calibrate:
            C = len(self._classes)

            def predict_fold(train, test):
                P = self._probs(X[test], X[train], self._y[train], self._w[train], C)
                return [_top({self._classes[j]: float(P[r, j]) for j in range(C)}, 3)
                        for r in range(len(test))]

            preds = cv_predictions(usable, predict_fold, self.folds, self.seed)
            if preds:
                calibration = choose_thresholds(preds, self.target_precision)
                self.thresholds = (calibration["min_prob"], calibration["min_margin"])
                if not calibration["met_target"]:
                    logger.warning("[Classifier] embedding backend: no thresholds reach %.0f%% "
                                   "precision on cross-validation; it will rarely answer.",
                                   self.target_precision * 100)
            else:
                logger.warning("[Classifier] too little data to calibrate; using default thresholds.")

        sources: dict = {}
        for e in usable:
            sources[e.source] = sources.get(e.source, 0) + 1
        self.stats = TrainStats(
            examples=len(usable), classes=len(self._classes), features=int(X.shape[1]),
            seconds=time.perf_counter() - t0, sources=sources, backend="embedding",
            calibration=calibration)
        return self.stats

    # -- persistence ---------------------------------------------------

    def save(self, path: "str | Path") -> None:
        if not self.trained:
            raise RuntimeError("nothing to save: the classifier is not trained")
        np = _np()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({
            "version": EMBEDDING_VERSION, "classes": self._classes, "model_name": self.model_name,
            "temperature": self.temperature,
            "thresholds": list(self.thresholds) if self.thresholds else None,
            "stats": self.stats.__dict__,
        })
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, X=self._X, y=self._y, w=self._w, meta=np.array(meta))
        tmp.replace(path)

    @classmethod
    def load(cls, path: "str | Path", embedder: Optional[Embedder] = None,
             device: str = "cpu", **_ignored) -> "EmbeddingIntentClassifier":
        """The model name is taken from the file: queries must be embedded by the
        same model the stored vectors came from. Raises on a bad file or when
        the embedding library is missing (the service catches)."""
        np = _np()
        with np.load(str(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("version") != EMBEDDING_VERSION:
                raise ValueError(f"unsupported embedding model version {meta.get('version')!r}")
            X, y, w = z["X"], z["y"], z["w"]
        clf = cls(embedder=embedder, model_name=meta["model_name"], device=device,
                  temperature=float(meta.get("temperature", 0.05)))
        if X.shape[0] != y.shape[0] or X.shape[0] != w.shape[0]:
            raise ValueError("embedding model file is internally inconsistent")
        clf._X, clf._y, clf._w = X.astype(np.float32), y.astype(np.int64), w.astype(np.float32)
        clf._classes = list(meta["classes"])
        th = meta.get("thresholds")
        clf.thresholds = (float(th[0]), float(th[1])) if th else None
        st = meta.get("stats") or {}
        clf.stats = TrainStats(**{k: st[k] for k in TrainStats.__dataclass_fields__ if k in st})
        if clf._embedder is None:                    # fail now, not on the first query
            clf._embedder = SentenceTransformerEmbedder(clf.model_name, device)
        return clf


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def _siblings(path: Path) -> tuple[Path, Path]:
    base = path.with_suffix("")
    return Path(f"{base}.emb.npz"), Path(f"{base}.ensemble.json")


class EnsembleClassifier:
    """TF-IDF and embedding probabilities, averaged. Saved as three files next
    to ``model_path``: the TF-IDF model itself, ``*.emb.npz`` and ``*.ensemble.json``."""

    def __init__(
        self,
        embedding: Optional[EmbeddingIntentClassifier] = None,
        tfidf: Optional[IntentClassifier] = None,
        weights: tuple[float, float] = (0.5, 0.5),
        folds: int = 5,
        target_precision: float = 0.95,
        tfidf_kwargs: Optional[dict] = None,
        seed: int = 0,
    ) -> None:
        self.embedding = embedding or EmbeddingIntentClassifier()
        self.tfidf = tfidf or IntentClassifier(**(tfidf_kwargs or {}))
        self.weights = weights
        self.folds = folds
        self.target_precision = target_precision
        self._tfidf_kwargs = tfidf_kwargs or {}
        self.seed = seed
        self.thresholds: Optional[tuple[float, float]] = None
        self.stats = TrainStats(backend="ensemble")

    @property
    def trained(self) -> bool:
        return self.tfidf.trained and self.embedding.trained

    @property
    def classes(self) -> list[str]:
        return self.embedding.classes

    @property
    def default_thresholds(self) -> tuple[float, float]:
        return self.thresholds or UNCALIBRATED_THRESHOLDS

    def _combine(self, emb: dict, tf: dict) -> dict:
        we, wt = self.weights
        return {c: we * emb.get(c, 0.0) + wt * tf.get(c, 0.0) for c in set(emb) | set(tf)}

    def predict(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        if not self.trained or not (text or "").strip():
            return []
        emb = self.embedding.class_probabilities(text)
        tf = dict(self.tfidf.predict(text, k=max(len(self.classes), 1)))
        return _top(self._combine(emb, tf), k)

    def classify(self, text: str, min_prob: Optional[float] = None,
                 min_margin: Optional[float] = None) -> Optional[Prediction]:
        d_prob, d_margin = self.default_thresholds
        return decide(self.predict(text, k=3),
                      d_prob if min_prob is None else min_prob,
                      d_margin if min_margin is None else min_margin)

    def fit(self, examples: Sequence[Example], calibrate: bool = True) -> TrainStats:
        t0 = time.perf_counter()
        usable = EmbeddingIntentClassifier.usable(examples)
        tf_stats = self.tfidf.fit(usable)
        self.embedding.fit(usable, calibrate=False)
        X, classes = self.embedding._X, self.embedding._classes
        C = len(classes)

        calibration: dict = {}
        self.thresholds = None
        if calibrate:
            def predict_fold(train, test):
                tf = IntentClassifier(**self._tfidf_kwargs)
                tf.fit([usable[i] for i in train])
                P = self.embedding._probs(X[test], X[train], self.embedding._y[train],
                                          self.embedding._w[train], C)
                tops = []
                for r, i in enumerate(test):
                    emb = {classes[j]: float(P[r, j]) for j in range(C)}
                    tops.append(_top(self._combine(emb, dict(tf.predict(usable[i].text, k=C))), 3))
                return tops

            preds = cv_predictions(usable, predict_fold, self.folds, self.seed)
            if preds:
                calibration = choose_thresholds(preds, self.target_precision)
                self.thresholds = (calibration["min_prob"], calibration["min_margin"])
            else:
                logger.warning("[Classifier] too little data to calibrate; using default thresholds.")
        self.stats = TrainStats(
            examples=len(usable), classes=C, features=tf_stats.features, epochs=tf_stats.epochs,
            train_accuracy=tf_stats.train_accuracy, seconds=time.perf_counter() - t0,
            sources=dict(tf_stats.sources), backend="ensemble", calibration=calibration)
        return self.stats

    def save(self, path: "str | Path") -> None:
        if not self.trained:
            raise RuntimeError("nothing to save: the classifier is not trained")
        path = Path(path)
        emb_path, meta_path = _siblings(path)
        self.tfidf.save(path)
        self.embedding.save(emb_path)
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "version": ENSEMBLE_VERSION, "weights": list(self.weights),
            "thresholds": list(self.thresholds) if self.thresholds else None,
            "stats": self.stats.__dict__,
        }), encoding="utf-8")
        tmp.replace(meta_path)

    @classmethod
    def load(cls, path: "str | Path", embedder: Optional[Embedder] = None,
             device: str = "cpu", **_ignored) -> "EnsembleClassifier":
        path = Path(path)
        emb_path, meta_path = _siblings(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != ENSEMBLE_VERSION:
            raise ValueError(f"unsupported ensemble model version {meta.get('version')!r}")
        ens = cls(embedding=EmbeddingIntentClassifier.load(emb_path, embedder=embedder, device=device),
                  tfidf=IntentClassifier.load(path), weights=tuple(meta.get("weights", (0.5, 0.5))))
        th = meta.get("thresholds")
        ens.thresholds = (float(th[0]), float(th[1])) if th else None
        st = meta.get("stats") or {}
        ens.stats = TrainStats(**{k: st[k] for k in TrainStats.__dataclass_fields__ if k in st})
        return ens


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_classifier(backend: str, embedding_model: Optional[str] = None, device: str = "cpu",
                    embedder: Optional[Embedder] = None, **_ignored):
    """An untrained classifier for *backend* (never the transformer: that one is
    produced by its own script)."""
    name = embedding_model or DEFAULT_EMBEDDING_MODEL
    if backend == "tfidf":
        return IntentClassifier()
    if backend == "embedding":
        return EmbeddingIntentClassifier(embedder=embedder, model_name=name, device=device)
    if backend == "ensemble":
        return EnsembleClassifier(embedding=EmbeddingIntentClassifier(
            embedder=embedder, model_name=name, device=device))
    raise ValueError(f"backend {backend!r} cannot be trained in-process")


def load_classifier(backend: str, path: "str | Path", embedding_model: Optional[str] = None,
                    device: str = "cpu", embedder: Optional[Embedder] = None, **_ignored):
    """Load a saved model for *backend*. Raises if the file is bad or a needed
    library is missing; ``ClassifierService.load`` catches and reports."""
    if backend == "tfidf":
        return IntentClassifier.load(path)
    if backend == "embedding":
        return EmbeddingIntentClassifier.load(path, embedder=embedder, device=device)
    if backend == "ensemble":
        return EnsembleClassifier.load(path, embedder=embedder, device=device)
    if backend == "transformer":
        from core.transformer_classifier import TransformerIntentClassifier
        return TransformerIntentClassifier.load(path, device=device)
    raise ValueError(f"unknown classifier backend {backend!r}")
