# tests/test_classifier_backends.py
"""
Backlog #25: augmentation, the embedding / ensemble / fine-tuned backends,
threshold calibration, the promotion gate, and the bench / fine-tune scripts.

No network, no model download, no torch. Embeddings come from a small
deterministic stand-in (hashed word counts) and the fine-tuned model from a
fake scorer, so these tests prove the MECHANICS (training, calibration, saving,
loading, gating, leak guards). They say nothing about how accurate the real
all-MiniLM-L6-v2 or distilbert models are on your queries; that is what
scripts/classifier_bench.py and scripts/finetune_classifier.py measure.
"""
import json
import os
import random
import sys
import zlib
from collections import namedtuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core import classifier_data as cd
from core.classifier_augment import AUGMENT_SOURCE, augment_examples, opener, tail, typo
from core.classifier_backends import (
    BACKENDS, EmbeddingIntentClassifier, EnsembleClassifier, choose_thresholds, cv_predictions,
    load_classifier, make_classifier,
)
from core.intent_classifier import ClassifierService, Example, IntentClassifier, TrainStats, decide
from core.transformer_classifier import TransformerIntentClassifier, write_labels


def fake_embedder(texts, d=128):
    """Hashed word counts: similar wording -> similar vectors. Not a real model."""
    out = np.zeros((len(texts), d), dtype=np.float32)
    for r, t in enumerate(texts):
        for w in str(t).lower().split():
            out[r, zlib.crc32(w.encode()) % d] += 1.0
    return out


SLEEP = ["log my sleep", "i slept for seven hours", "track my sleep last night", "record sleep of eight hours",
         "note that i slept six hours"]
WORKOUT = ["log my workout", "i worked out today", "record my workout please", "track my gym session",
           "note a run of five km"]
WATER = ["log a glass of water", "i drank water", "record my water intake", "track water i drank",
         "add a cup of water"]


def tiny():
    return ([Example(t, "apollo_track_sleep") for t in SLEEP]
            + [Example(t, "apollo_log_workout") for t in WORKOUT]
            + [Example(t, "apollo_log_water") for t in WATER])


TINY_TFIDF = {"epochs": 25}


# ------------------------------------------------------------------ augmentation

def test_augmentation_is_deterministic_and_marks_its_output():
    base = [Example("create a goal to finish the project", "artemis_add_goal"),
            Example("add goal launch website", "artemis_add_goal")]
    a = augment_examples(base, target_per_intent=6, seed=3)
    b = augment_examples(base, target_per_intent=6, seed=3)
    assert [e.text for e in a] == [e.text for e in b] and a
    assert all(e.source == AUGMENT_SOURCE and e.weight < 1.0 and e.parent for e in a)
    assert {e.intent for e in a} == {"artemis_add_goal"}
    assert len(base) + len(a) <= 6


def test_intents_with_enough_examples_are_left_alone():
    assert augment_examples(tiny(), target_per_intent=5) == []


def test_augmentation_never_reproduces_existing_or_forbidden_text():
    base = [Example("remind me to call mom", "chronos_set_reminder")]
    forbidden = ["please remind me to call mom", "remind me to call mom now"]
    out = augment_examples(base, target_per_intent=12, forbidden=forbidden)
    keys = {e.text.lower() for e in out}
    assert "remind me to call mom" not in keys
    assert not keys & {f.lower() for f in forbidden}
    assert len(keys) == len(out)


def test_transforms_leave_digits_and_short_words_alone_and_skip_redundant_edits():
    rng = random.Random(0)
    assert typo("log 500 now at 9", rng) is None                    # nothing long enough to garble
    assert typo("remind me", rng) != "remind me"
    assert opener("please log my sleep", rng) is None
    assert tail("log my sleep please", rng) is None
    assert "500" in (tail("log 500", rng) or "")


def test_golden_prompts_never_enter_training_even_with_augmentation():
    held = cd.held_out_queries()
    ex = cd.collect_examples(augment=True)
    assert held, "the golden file should be present in the repo"
    assert not {cd.normalise(e.text) for e in ex} & held
    assert any(e.source == AUGMENT_SOURCE for e in ex)


def test_augment_flag_is_off_by_default():
    assert not any(e.source == AUGMENT_SOURCE for e in cd.collect_examples())


# ------------------------------------------------------------------ thresholds and CV

def test_choose_thresholds_picks_the_loosest_pair_that_meets_the_target():
    good = [("a", [("a", 0.9), ("b", 0.05)])] * 20                # confident and right
    shaky = [("b", [("a", 0.45), ("b", 0.40)])] * 10              # unsure and wrong
    res = choose_thresholds(good + shaky, target_precision=0.95)
    assert res["met_target"] and res["precision"] >= 0.95
    assert decide(good[0][1], res["min_prob"], res["min_margin"]) is not None
    assert decide(shaky[0][1], res["min_prob"], res["min_margin"]) is None
    assert res["coverage"] == pytest.approx(20 / 30, abs=0.01)


def test_choose_thresholds_reports_when_nothing_qualifies():
    bad = [("a", [("b", 0.9), ("a", 0.05)])] * 30
    res = choose_thresholds(bad, target_precision=0.95)
    assert res["met_target"] is False and res["coverage"] == 0.0


def test_cv_keeps_variants_with_their_parents_and_scores_only_real_examples():
    base = tiny()
    examples = base + augment_examples(base[:2], target_per_intent=8)
    seen = []

    def predict_fold(train, test):
        for i in test:
            key = " ".join(examples[i].text.lower().split())
            assert examples[i].source != AUGMENT_SOURCE
            assert not any(examples[j].parent == key for j in train)   # no variant of a test example in training
        seen.extend(test)
        return [[(examples[i].intent, 0.9)] for i in test]

    preds = cv_predictions(examples, predict_fold, folds=4, seed=1)
    assert preds and len(preds) == len(seen)
    assert all(truth == top[0][0] for truth, top in preds)


def test_cv_with_too_little_data_returns_nothing():
    assert cv_predictions([Example("a", "x"), Example("b", "y")], lambda t, s: [], folds=5) == []


def test_decide_is_the_one_shared_rule():
    assert decide([], 0.1, 0.1) is None
    assert decide([("a", 0.9)], 0.5, 0.2).intent == "a"
    assert decide([("a", 0.5), ("b", 0.45)], 0.3, 0.1) is None           # lead too small
    assert decide([("a", 0.2), ("b", 0.01)], 0.3, 0.1) is None           # too unsure


# ------------------------------------------------------------------ embedding backend

def test_embedding_classifier_trains_calibrates_and_answers():
    clf = EmbeddingIntentClassifier(embedder=fake_embedder, folds=3)
    stats = clf.fit(tiny())
    assert clf.trained and stats.backend == "embedding" and stats.classes == 3 and stats.features == 128
    assert clf.thresholds is not None and stats.calibration["scored"] > 0
    assert clf.predict("log my sleep", k=3)[0][0] == "apollo_track_sleep"
    assert abs(sum(p for _, p in clf.predict("x log", k=10)) - 1) < 1e-4
    assert clf.classify("i slept for seven hours").intent == "apollo_track_sleep"


def test_embedding_classifier_declines_gibberish_and_empty():
    clf = EmbeddingIntentClassifier(embedder=fake_embedder, folds=3)
    clf.fit(tiny())
    assert clf.predict("", k=3) == [] and clf.classify("   ") is None
    assert clf.classify("zzzz qqqq xxxx", min_prob=0.95, min_margin=0.5) is None


def test_embedding_classifier_needs_two_intents_and_valid_shapes():
    with pytest.raises(ValueError):
        EmbeddingIntentClassifier(embedder=fake_embedder).fit([Example("a b", "x"), Example("c d", "x")])
    with pytest.raises(ValueError):
        EmbeddingIntentClassifier(embedder=lambda t: np.zeros((1, 4))).fit(tiny())


def test_embedding_classifier_saves_and_loads_without_pickle(tmp_path):
    clf = EmbeddingIntentClassifier(embedder=fake_embedder, folds=3, model_name="my-model")
    clf.fit(tiny())
    path = tmp_path / "e.npz"
    clf.save(path)
    again = EmbeddingIntentClassifier.load(path, embedder=fake_embedder)
    assert again.model_name == "my-model" and again.thresholds == clf.thresholds
    assert again.classes == clf.classes and again.stats.examples == clf.stats.examples
    assert again.predict("log my sleep")[0][0] == "apollo_track_sleep"
    with np.load(str(path), allow_pickle=False):                  # loads with pickle disabled
        pass


def test_loading_a_bad_embedding_file_raises(tmp_path):
    p = tmp_path / "bad.npz"
    p.write_bytes(b"not a model")
    with pytest.raises(Exception):
        EmbeddingIntentClassifier.load(p, embedder=fake_embedder)


def test_loading_without_the_embedding_library_fails_loudly(tmp_path, monkeypatch):
    clf = EmbeddingIntentClassifier(embedder=fake_embedder, folds=3)
    clf.fit(tiny())
    clf.save(tmp_path / "e.npz")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)       # import now fails
    with pytest.raises(Exception):
        EmbeddingIntentClassifier.load(tmp_path / "e.npz")


def test_augmentation_reaches_the_embedding_model_with_lower_weight():
    base = tiny()
    clf = EmbeddingIntentClassifier(embedder=fake_embedder, folds=3)
    clf.fit(base + augment_examples(base[:1], target_per_intent=9))
    assert clf.stats.sources.get(AUGMENT_SOURCE, 0) > 0
    assert float(clf._w.min()) < 1.0 <= float(clf._w.max())


# ------------------------------------------------------------------ ensemble

def _ensemble():
    return EnsembleClassifier(embedding=EmbeddingIntentClassifier(embedder=fake_embedder, folds=3),
                              folds=3, tfidf_kwargs=TINY_TFIDF)


def test_ensemble_trains_both_members_and_answers():
    ens = _ensemble()
    stats = ens.fit(tiny())
    assert ens.trained and stats.backend == "ensemble" and stats.classes == 3
    assert ens.classify("track my gym session").intent == "apollo_log_workout"
    assert abs(sum(p for _, p in ens.predict("log water", k=10)) - 1) < 0.05
    assert ens.predict("", k=3) == []


def test_ensemble_saves_three_files_and_reloads(tmp_path):
    ens = _ensemble()
    ens.fit(tiny())
    path = tmp_path / "m.npz"
    ens.save(path)
    assert path.is_file() and (tmp_path / "m.emb.npz").is_file() and (tmp_path / "m.ensemble.json").is_file()
    again = EnsembleClassifier.load(path, embedder=fake_embedder)
    assert again.thresholds == ens.thresholds and again.weights == ens.weights
    assert again.classify("i drank water").intent == "apollo_log_water"


def test_ensemble_with_a_missing_member_file_does_not_load(tmp_path):
    ens = _ensemble()
    ens.fit(tiny())
    ens.save(tmp_path / "m.npz")
    (tmp_path / "m.emb.npz").unlink()
    with pytest.raises(Exception):
        EnsembleClassifier.load(tmp_path / "m.npz", embedder=fake_embedder)


# ------------------------------------------------------------------ the service

def _svc(tmp_path, backend, **kw):
    return ClassifierService(model_path=tmp_path / "m.npz", mode="assist", backend=backend,
                             examples_fn=kw.pop("examples_fn", tiny), embedder=fake_embedder, **kw)


def test_service_trains_saves_and_reloads_the_embedding_backend(tmp_path):
    svc = _svc(tmp_path, "embedding")
    assert svc.train()["ok"] and svc.status()["backend"] == "embedding"
    assert svc.status()["thresholds"] is not None
    assert svc.classify("i slept for seven hours").intent == "apollo_track_sleep"
    fresh = _svc(tmp_path, "embedding", examples_fn=None)
    assert fresh.load() and fresh.classify("log my workout").intent == "apollo_log_workout"


def test_service_ensemble_roundtrip(tmp_path):
    svc = _svc(tmp_path, "ensemble")
    assert svc.train()["ok"]
    assert (tmp_path / "m.ensemble.json").is_file()
    fresh = ClassifierService(model_path=tmp_path / "m.npz", mode="assist", backend="ensemble",
                              embedder=fake_embedder)
    assert fresh.load() and fresh.ready


def test_configured_thresholds_override_the_models_own(tmp_path):
    svc = _svc(tmp_path, "embedding", min_prob=0.999, min_margin=0.999)
    svc.train()
    assert svc.classify("log my sleep") is None
    svc.min_prob = svc.min_margin = None
    assert svc.classify("log my sleep") is not None


def test_unknown_backend_falls_back_to_tfidf(tmp_path):
    assert ClassifierService(model_path=tmp_path / "m.npz", backend="nonsense").backend == "tfidf"
    assert set(BACKENDS) == {"tfidf", "embedding", "ensemble", "transformer"}


def test_default_service_is_unchanged_tfidf(tmp_path):
    svc = ClassifierService(model_path=tmp_path / "m.npz", examples_fn=tiny)
    assert svc.backend == "tfidf" and svc.train()["ok"]
    assert svc.status()["thresholds"] == [0.30, 0.12]
    assert isinstance(svc._clf, IntentClassifier)


def test_transformer_backend_is_never_trained_in_process(tmp_path):
    svc = ClassifierService(model_path=tmp_path / "dir", mode="assist", backend="transformer",
                            examples_fn=tiny)
    assert svc.trainable is False
    assert "finetune_classifier" in svc.train()["error"]
    assert svc.retrain_if_needed()["skipped"]
    svc.ensure_ready(background=True)                              # must not start a training thread
    assert not svc.ready and "finetune_classifier" in svc.last_error
    assert svc.classify("anything") is None


def test_make_classifier_refuses_the_transformer():
    with pytest.raises(ValueError):
        make_classifier("transformer")
    with pytest.raises(ValueError):
        load_classifier("nope", "x")


# ------------------------------------------------------------------ fine-tuned wrapper

CLASSES = ["apollo_log_workout", "apollo_track_sleep"]


def keyword_scorer(texts):
    out = np.zeros((len(texts), 2))
    for r, t in enumerate(texts):
        out[r, 1 if "sleep" in t else 0] = 6.0 if ("sleep" in t or "workout" in t) else 0.0
    return out


def test_transformer_wrapper_softmaxes_and_declines():
    clf = TransformerIntentClassifier(keyword_scorer, CLASSES, thresholds=(0.6, 0.2))
    top = clf.predict("log my sleep", k=2)
    assert top[0][0] == "apollo_track_sleep" and top[0][1] > 0.99
    assert clf.classify("log my sleep").intent == "apollo_track_sleep"
    assert clf.classify("something else") is None                  # equal logits: no margin
    assert clf.predict("  ") == []


def test_transformer_wrapper_rejects_a_mismatched_output():
    clf = TransformerIntentClassifier(lambda t: np.zeros((1, 5)), CLASSES)
    with pytest.raises(ValueError):
        clf.predict("x")
    svc = ClassifierService(model_path="x", mode="assist", backend="transformer")
    svc._clf = clf
    assert svc.classify("x") is None                               # the service swallows it


def test_transformer_labels_roundtrip(tmp_path):
    write_labels(tmp_path, CLASSES, (0.5, 0.1), "distilbert-base-uncased", TrainStats(examples=9))
    loaded = TransformerIntentClassifier.load(tmp_path, scorer=keyword_scorer)
    assert loaded.classes == CLASSES and loaded.thresholds == (0.5, 0.1) and loaded.stats.examples == 9
    assert loaded.stats.backend == "transformer"
    with pytest.raises(Exception):
        TransformerIntentClassifier.load(tmp_path / "missing", scorer=keyword_scorer)


def test_service_reports_a_load_error_instead_of_raising_when_torch_is_missing(tmp_path):
    write_labels(tmp_path, CLASSES, None, "x", TrainStats())
    svc = ClassifierService(model_path=tmp_path, mode="assist", backend="transformer")
    ok = svc.load()
    assert ok or "couldn't load" in svc.last_error      # no torch here -> a reported error, never an exception


# ------------------------------------------------------------------ gate

def _r(cov, prec, n=99):
    return {"cases": n, "coverage": cov, "precision": prec}


def test_gate_requires_a_real_gain_without_losing_precision():
    ok, why = cd.gate_decision(_r(0.58, 0.95), _r(0.64, 0.94))
    assert ok and "coverage" in why[0]
    assert not cd.gate_decision(_r(0.58, 0.95), _r(0.60, 0.95))[0]          # gain too small
    assert not cd.gate_decision(_r(0.58, 0.95), _r(0.70, 0.92))[0]          # precision fell 3 points
    assert not cd.gate_decision(_r(0.58, 0.91), _r(0.70, 0.89))[0]          # under the floor
    ok, why = cd.gate_decision(_r(0.5, 0.9, n=20), _r(0.9, 0.99, n=20))
    assert not ok and "comparable golden cases" in why[0]
    assert not cd.gate_decision(_r(0.5, 0.9, n=99), _r(0.9, 0.99, n=80))[0]  # different case sets


def test_wilson_interval_is_wide_for_small_samples():
    lo, hi = cd.wilson_interval(54, 57)
    assert lo < 0.90 < 0.947 < hi <= 1.0
    assert cd.wilson_interval(0, 0) == (0.0, 0.0)


# ------------------------------------------------------------------ fine-tune script (fakes, no torch)

Case = namedtuple("Case", "prompt kind expected forbidden")

NAMES = {"apollo_track_sleep": "sleep", "apollo_log_workout": "workout",
         "apollo_log_water": "water", "apollo_log_meal": "meal"}
CODES = {"apollo_track_sleep": "zzs", "apollo_log_workout": "zzw",
         "apollo_log_water": "zzr", "apollo_log_meal": "zzm"}


def _ft_world(quality):
    """4 intents x 6 training examples, and 60 golden cases written with letters
    that never occur in training, so TF-IDF has nothing to match and declines
    them all. A 'good' fake model recognises both the training words and the
    golden-case codes; a 'useless' one recognises nothing."""
    examples = [Example(f"{w} number {i} log it", intent) for intent, w in NAMES.items() for i in range(6)]
    cases = [Case(f"qqq{i} {list(CODES.values())[i % 4]}", "intent", list(NAMES)[i % 4], None)
             for i in range(60)]
    classes = sorted(NAMES)

    def train_fn(train, val, cand, **kw):
        cand.mkdir(parents=True, exist_ok=True)
        (cand / "weights.bin").write_text("fake")
        return {"classes": classes, "best_val_accuracy": 0.9, "epochs_run": 1, "device": "cpu", "seconds": 0.1}

    def scorer_fn(cand, device):
        def scorer(texts):
            out = np.zeros((len(texts), len(classes)))
            if quality == "good":
                for r, t in enumerate(texts):
                    for j, c in enumerate(classes):
                        if NAMES[c] in t or CODES[c] in t:
                            out[r, j] = 8.0
            return out
        return scorer
    return examples, cases, train_fn, scorer_fn


def _run_ft(tmp_path, quality, promote=True):
    from scripts import finetune_classifier as ft
    examples, cases, train_fn, scorer_fn = _ft_world(quality)
    lines = []
    out_dir = tmp_path / "model"
    code = ft.run_finetune(base_model="fake", epochs=1, batch_size=4, device="cpu", out_dir=out_dir,
                           augment=False, promote=promote, report_path=tmp_path / "report.json",
                           out=lines.append, train_fn=train_fn, scorer_fn=scorer_fn,
                           examples=examples, cases=cases)
    return code, out_dir, json.loads((tmp_path / "report.json").read_text()), "\n".join(lines)


def test_a_clearly_better_model_is_promoted(tmp_path):
    code, out_dir, report, log = _run_ft(tmp_path, "good")
    assert report["baseline"]["answered"] == 0                      # TF-IDF really has nothing to go on
    assert code == 0 and report["gate_passed"] and report["promoted"], log
    assert (out_dir / "weights.bin").is_file() and (out_dir / "labels.json").is_file()
    assert not (tmp_path / "model.candidate").exists()
    assert report["candidate"]["coverage"] > report["baseline"]["coverage"]
    assert "PROMOTED" in log


def test_a_useless_model_is_rejected_and_left_as_a_candidate(tmp_path):
    code, out_dir, report, log = _run_ft(tmp_path, "useless")
    assert code == 3 and not report["gate_passed"] and not report["promoted"]
    assert not out_dir.exists() and (tmp_path / "model.candidate" / "labels.json").is_file()
    assert "REJECTED" in log and report["gate_reasons"]


def test_no_promote_never_installs_even_a_passing_model(tmp_path):
    code, out_dir, report, _ = _run_ft(tmp_path, "good", promote=False)
    assert code == 0 and report["gate_passed"] and not report["promoted"] and not out_dir.exists()


def test_promotion_keeps_the_previous_model(tmp_path):
    out_dir = tmp_path / "model"
    out_dir.mkdir()
    (out_dir / "old.txt").write_text("previous")
    _run_ft(tmp_path, "good")
    assert (tmp_path / "model.previous" / "old.txt").read_text() == "previous"


def test_validation_split_never_leaks_into_training():
    from scripts.finetune_classifier import split_train_val
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    base = ([Example(f"intent a {w}", "a") for w in words] + [Example(f"intent b {w}", "b") for w in words])
    train, val = split_train_val(base + augment_examples(base, target_per_intent=14), frac=0.3, seed=2)
    assert val and all(v.source != AUGMENT_SOURCE for v in val)
    val_keys = {" ".join(v.text.lower().split()) for v in val}
    assert not any(t.parent in val_keys for t in train)                # no variant of a val example
    assert not val_keys & {" ".join(t.text.lower().split()) for t in train}


def test_split_keeps_rare_intents_entirely_in_training():
    from scripts.finetune_classifier import split_train_val
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    ex = [Example("only one", "rare"), Example("only two", "rare")] + [Example(f"common {w}", "c") for w in words]
    train, val = split_train_val(ex, frac=0.5)
    assert val and all(v.intent == "c" for v in val) and sum(e.intent == "rare" for e in train) == 2


def test_finetune_main_reports_missing_dependencies(monkeypatch):
    from scripts import finetune_classifier as ft
    monkeypatch.setattr(ft, "check_dependencies", lambda: "missing: torch")
    assert ft.main([]) == 2


# ------------------------------------------------------------------ bench script

def test_bench_row_for_an_embedding_backend_on_small_data():
    from scripts import classifier_bench as cb
    cases = [Case("log my sleep", "intent", "apollo_track_sleep", None),
             Case("track my gym session", "intent", "apollo_log_workout", None)]
    row = cb.run_one("embedding", False, embedder=fake_embedder, examples=tiny(), cases=cases)
    assert row["backend"] == "embedding" and row["cases"] == 2 and "precision_ci" in row
    assert "answered" in cb.format_row(row) and "embedding" in cb.format_row(row)


def test_bench_reports_a_failed_backend_instead_of_raising():
    from scripts import classifier_bench as cb
    row = cb.run_one("embedding", False, embedder=fake_embedder, examples=[Example("a", "x")], cases=[])
    assert "error" in row and "FAILED" in cb.format_row(row)


def test_bench_main_rejects_the_transformer_and_saves_a_baseline(tmp_path, monkeypatch):
    from scripts import classifier_bench as cb
    lines = []
    assert cb.main(["--backends", "transformer"], out=lines.append) == 2
    fake_row = {"backend": "tfidf", "augment": False, "examples": 5, "classes": 2, "cases": 9, "answered": 5,
                "right": 5, "coverage": 0.55, "precision": 1.0, "precision_ci": [0.5, 1.0],
                "thresholds": [0.3, 0.12], "train_seconds": 0.1, "calibration": {}, "wrong": []}
    monkeypatch.setattr(cb, "run_one", lambda backend, *a, **k: dict(fake_row, backend=backend))
    path = tmp_path / "baseline.json"
    assert cb.main(["--save-baseline"], out=lines.append, baseline_path=path) == 0
    saved = json.loads(path.read_text())
    assert saved["coverage"] == 0.55 and saved["recorded"]
    lines.clear()
    assert cb.main([], out=lines.append, baseline_path=path) == 0
    assert any("Saved baseline" in l for l in lines)
    assert cb.main(["--backends", "embedding", "--save-baseline"], out=lines.append, baseline_path=path) == 2


# ------------------------------------------------------------------ config plumbing

def test_builder_classifier_kwargs_defaults_and_overrides():
    from main import HestiaBuilder
    k = HestiaBuilder._classifier_kwargs({}, "assist")
    assert k["backend"] == "tfidf" and k["model_path"] == "data/intent_classifier.npz"
    assert k["min_prob"] is None and k["min_margin"] is None
    k = HestiaBuilder._classifier_kwargs({"backend": "Embedding", "min_probability": 0.4,
                                          "augment": True, "device": "cuda"}, "primary")
    assert k["backend"] == "embedding" and k["min_prob"] == 0.4 and k["device"] == "cuda"
    assert any(e.source == AUGMENT_SOURCE for e in k["examples_fn"]())


def test_builder_handles_bad_backend_and_transformer_paths():
    from main import HestiaBuilder
    assert HestiaBuilder._classifier_kwargs({"backend": "magic"}, "assist")["backend"] == "tfidf"
    k = HestiaBuilder._classifier_kwargs({"backend": "transformer", "model_path": "data/intent_classifier.npz"}, "assist")
    assert k["model_path"] == "data/intent_distilbert"
    assert HestiaBuilder._classifier_kwargs({"backend": "transformer", "model_path": "data/mine"}, "assist")["model_path"] == "data/mine"


def test_config_validation_knows_the_new_keys():
    from core.config_validation import validate_config
    base = {"ollama": {"model": "m"}, "database": {"path": "x.db"}}
    good = validate_config({**base, "classifier": {"mode": "off", "backend": "ensemble", "augment": True,
                                                   "augment_target": 8, "embedding_model": "m", "device": "cpu"}})
    bad = validate_config({**base, "classifier": {"augment": "yes", "backend": 3}})
    assert good.ok and not any("classifier" in w for w in good.warnings)
    assert not bad.ok
    text = " ".join(bad.errors)
    assert "classifier.augment" in text and "classifier.backend" in text
