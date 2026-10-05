# tests/test_iris_video_faces_detection.py
"""
Tests for Iris backlog #71 (re-index, find-similar, relevance cut-offs),
#72 (local face grouping), #75 (video) and #77 (object detection).
No real models are used: OpenCV is used only for synthetic videos (skipped if
absent); embedders, face backends, detectors and cameras are fakes.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.iris import detection, faces, video
from modules.iris.db import IrisDB
from modules.iris.embeddings import filter_hits
from modules.iris.iris_engine import IrisEngine


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_db(tmp_path):
    return IrisDB(str(tmp_path / "iris.db"))


def add_file(db, path, ftype="image", h=None):
    return db.insert_file(str(path), h or f"hash-{path}", None, 10, ftype, "x/y")


class FakeIndex:
    available = True

    def __init__(self):
        self.vecs = {}

    def upsert(self, fid, vec):
        self.vecs[fid] = vec
        return True

    def indexed_ids(self):
        return set(self.vecs)

    def get_embedding(self, fid):
        return self.vecs.get(fid)

    def query(self, vec, top_k=10):
        def dist(a, b):
            return 1.0 - faces.cosine(a, b)
        hits = sorted(((fid, dist(vec, v)) for fid, v in self.vecs.items()), key=lambda h: h[1])
        return hits[:top_k]


class FakeEmbedder:
    def __init__(self, vec=(1.0, 0.0)):
        self.vec = list(vec)

    def embed_text(self, text):
        return self.vec

    def embed_image(self, path):
        return self.vec


def make_engine(tmp_path, **kw):
    engine = IrisEngine.__new__(IrisEngine)
    engine.db = make_db(tmp_path)
    engine.config = type("C", (), {"cache_dir": str(tmp_path), "video_frames": 4,
                                   "detector_confidence": 0.4, "camera_enabled": kw.pop("camera", False),
                                   "semantic_max_distance": None, "semantic_relative_margin": None})()
    engine.embedder = kw.pop("embedder", FakeEmbedder())
    engine.vector_index = kw.pop("vector_index", FakeIndex())
    engine.faces = kw.pop("face_service", faces.FaceService(engine.db, None, enabled=False))
    engine.detector = kw.pop("detector", None)
    engine._camera_factory = kw.pop("camera_factory", None)
    return engine


# ---------------------------------------------------------------------------
# #71 — filter_hits, find_similar, reindex
# ---------------------------------------------------------------------------

def test_filter_hits_noop_by_default():
    hits = [(1, 0.1), (2, 0.9)]
    assert filter_hits(hits) == hits


def test_filter_hits_max_distance():
    assert filter_hits([(1, 0.1), (2, 0.5), (3, 0.9)], max_distance=0.5) == [(1, 0.1), (2, 0.5)]


def test_filter_hits_relative_margin():
    assert filter_hits([(1, 0.1), (2, 0.15), (3, 0.6)], relative_margin=0.1) == [(1, 0.1), (2, 0.15)]


def test_filter_hits_empty():
    assert filter_hits([], max_distance=0.1, relative_margin=0.1) == []


def test_find_similar_orders_nearest_first_and_excludes_self(tmp_path):
    e = make_engine(tmp_path)
    ids = [add_file(e.db, f"/p/{i}.jpg") for i in range(3)]
    e.vector_index.vecs = {ids[0]: [1.0, 0.0], ids[1]: [0.9, 0.1], ids[2]: [0.0, 1.0]}
    out = e.find_similar(ids[0])
    assert [m["id"] for m in out["matches"]] == [ids[1], ids[2]]
    assert "distance" in out["matches"][0]


def test_find_similar_unknown_photo(tmp_path):
    e = make_engine(tmp_path)
    assert e.find_similar(999)["error"]


def test_find_similar_no_embedding(tmp_path):
    e = make_engine(tmp_path)
    fid = add_file(e.db, "/missing/x.jpg")
    assert e.find_similar(fid)["error"]


def test_reindex_indexes_only_unindexed(tmp_path):
    e = make_engine(tmp_path)
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    for p in (p1, p2):
        p.write_bytes(b"x")
    f1, f2 = add_file(e.db, p1), add_file(e.db, p2)
    e.vector_index.vecs[f1] = [1.0, 0.0]
    stats = e.reindex_embeddings()
    assert stats["indexed"] == 1 and f2 in e.vector_index.vecs and stats["remaining"] == 0


def test_reindex_counts_unreadable_as_failed(tmp_path):
    e = make_engine(tmp_path)
    add_file(e.db, tmp_path / "gone.jpg")
    assert e.reindex_embeddings()["failed"] == 1


def test_reindex_batches(tmp_path):
    e = make_engine(tmp_path)
    for i in range(3):
        p = tmp_path / f"{i}.jpg"
        p.write_bytes(b"x")
        add_file(e.db, p)
    stats = e.reindex_embeddings(limit=2)
    assert stats["indexed"] == 2 and stats["remaining"] == 1


def test_reindex_unavailable_without_embedder(tmp_path):
    class Dead:
        def embed_text(self, t):
            return None
    e = make_engine(tmp_path, embedder=Dead())
    assert e.reindex_embeddings()["unavailable"] is True


# ---------------------------------------------------------------------------
# #75 — video
# ---------------------------------------------------------------------------

def test_sample_positions():
    assert video.sample_positions(50, 4) == [6, 18, 31, 43]
    assert video.sample_positions(2, 4) == [0, 1]
    assert video.sample_positions(0, 4) == []


def test_format_duration():
    assert video.format_duration(42) == "0:42"
    assert video.format_duration(3725) == "1:02:05"
    assert video.format_duration(None) == "" and video.format_duration(0) == ""


def test_mean_embedding_normalised_and_robust():
    assert video.mean_embedding([]) is None
    assert video.mean_embedding([[0.0, 0.0]]) is None
    m = video.mean_embedding([[1.0, 0.0], [0.0, 5.0]])
    assert abs(sum(x * x for x in m) - 1.0) < 1e-9
    assert abs(m[0] - m[1]) < 1e-9   # magnitude of the second vector must not dominate


def test_combine_frame_results():
    cap, tags, mood = video.combine_frame_results(
        [("a dog", "dog, park", "happy"), ("A dog", "park, tree", "happy"), ("a cat", "cat", "calm")])
    assert cap == "Video: a dog → a cat"
    assert tags[0] == "video" and "tree" in tags and tags.count("park") == 1
    assert mood == "happy"
    assert video.combine_frame_results([])[0] == "Unlabeled video"


def _write_video(path, frames=50):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not w.isOpened():
        pytest.skip("this OpenCV can't write MJPG")
    for i in range(frames):
        f = np.zeros((48, 64, 3), np.uint8)
        if i >= 10:
            f[:, :] = (i * 4 % 255, 100, 200 - i)
        w.write(f)
    w.release()


def test_probe_and_sample_real_video(tmp_path):
    p = tmp_path / "t.avi"
    _write_video(p)
    info = video.probe_video(p)
    assert info and info.frame_count == 50 and abs(info.duration_seconds - 5.0) < 0.01
    out = video.sample_frames(p, tmp_path / "o", 4)
    assert 1 <= len(out) <= 4 and all(f.exists() for f in out)


def test_sample_frames_bad_file_returns_empty(tmp_path):
    pytest.importorskip("cv2")
    p = tmp_path / "bad.mp4"
    p.write_bytes(b"not a video")
    assert video.sample_frames(p, tmp_path / "o") == []
    assert video.probe_video(p) is None


def test_analyse_video_end_to_end(tmp_path):
    from modules.iris.analyser import IrisAnalyser
    p = tmp_path / "t.avi"
    _write_video(p)
    db = make_db(tmp_path)
    fid = add_file(db, p, ftype="video")
    index = FakeIndex()
    a = IrisAnalyser(db, "h", 1, embedder=FakeEmbedder(), vector_index=index, work_dir=tmp_path)
    a._send_to_ollama = lambda img, prompt: "CAPTION: A colour test card.\nTAGS: test, colour\nMOOD: calm"
    assert a.analyse_file(fid) is True
    rec = db.get_file(fid)
    assert rec["caption"].startswith("Video:") and "video" in json.loads(rec["tags"])
    assert rec["processed"] == 1 and rec["duration_seconds"] > 0
    assert fid in index.vecs


def test_analyse_video_unreadable_marks_error(tmp_path):
    pytest.importorskip("cv2")
    from modules.iris.analyser import IrisAnalyser
    p = tmp_path / "bad.mp4"
    p.write_bytes(b"nope")
    db = make_db(tmp_path)
    fid = add_file(db, p, ftype="video")
    a = IrisAnalyser(db, "h", 1, work_dir=tmp_path)
    assert a.analyse_file(fid) is False
    assert db.get_file(fid)["analysis_status"] == "error"


def test_perceptual_hash_skipped_for_video(tmp_path):
    import asyncio
    from modules.iris.ingestion import DuplicateDetector
    d = DuplicateDetector(make_db(tmp_path))
    assert asyncio.run(d.compute_perceptual_hash(tmp_path / "x.mp4")) is None


def test_search_result_marks_videos(tmp_path):
    e = make_engine(tmp_path)
    fid = add_file(e.db, "/v/clip.mp4", ftype="video")
    e.db.update_video_info(fid, 42)
    e.db.update_file_analysis(fid, "Video: a beach", json.dumps(["video", "beach"]), None, "calm", False, None)
    e.db.mark_file_processed(fid)
    out = e.search("beach")
    assert "(video, 0:42)" in out


# ---------------------------------------------------------------------------
# #72 — faces
# ---------------------------------------------------------------------------

def _f(i, vec, score=0.9):
    return {"id": i, "embedding": vec, "score": score}


def test_group_faces_basic():
    r = faces.group_faces([_f(1, [1, 0]), _f(2, [.99, .1]), _f(3, [0, 1])], {}, 0.45, 2)
    assert r["new"] == [[1, 2]] and r["unclustered"] == [3] and r["existing"] == {}


def test_group_faces_joins_existing_person():
    r = faces.group_faces([_f(1, [1, 0])], {7: [1.0, 0.0]}, 0.45, 2)
    assert r["existing"] == {1: 7} and r["new"] == []


def test_group_faces_deterministic_regardless_of_input_order():
    items = [_f(i, [1, i * 0.01]) for i in range(1, 6)]
    assert faces.group_faces(items, {}, 0.45, 2) == faces.group_faces(list(reversed(items)), {}, 0.45, 2)


def test_group_faces_conservative_does_not_merge_distinct():
    r = faces.group_faces([_f(1, [1, 0]), _f(2, [1, 0]), _f(3, [0, 1]), _f(4, [0, 1])], {}, 0.45, 2)
    assert sorted(map(sorted, r["new"])) == [[1, 2], [3, 4]]


def test_clean_name():
    assert faces.clean_name("  Mom. ") == "Mom"
    assert faces.clean_name("") is None and faces.clean_name("x" * 100) is None and faces.clean_name(5) is None


class FakeBackend:
    unavailable_reason = None

    def __init__(self, table):
        self.table = table

    def extract(self, path):
        r = self.table[os.path.basename(path)]
        if isinstance(r, Exception):
            raise r
        return [faces.FaceDetection((0.1, 0.1, 0.2, 0.2), 0.9, v) for v in r]


def face_service(db, table, **kw):
    return faces.FaceService(db, FakeBackend(table), enabled=True, **kw)


def test_faces_off_by_default(tmp_path):
    db = make_db(tmp_path)
    svc = faces.FaceService(db, FakeBackend({}))
    out = svc.scan()
    assert out["available"] is False and "switched off" in out["message"]
    assert svc.person_in_query("photos of Mom") is None


def test_faces_unavailable_backend_reports_reason(tmp_path):
    class B:
        unavailable_reason = "no face models configured"
    out = faces.FaceService(make_db(tmp_path), B(), enabled=True).scan()
    assert out["available"] is False and "no face models" in out["message"]


def test_scan_group_name_find_and_forget(tmp_path):
    db = make_db(tmp_path)
    ids = [add_file(db, f"/p/{n}.jpg") for n in ("a", "b", "c", "d")]
    table = {"a.jpg": [[1, 0]], "b.jpg": [[.98, .1]], "c.jpg": [[0, 1]], "d.jpg": [[1, 0], [0, 1]]}
    svc = face_service(db, table)
    out = svc.scan()
    assert out["available"] and out["scanned"] == 4 and out["faces_found"] == 5
    assert out["new_people"] == 2 and out["remaining"] == 0
    # rescan does nothing new
    assert svc.scan()["scanned"] == 0
    people = svc.list_people()
    assert len(people) == 2 and all(p["label"].startswith("Person ") for p in people)

    pid = people[0]["id"]
    assert svc.name_person(f"Person {pid}", "Mom")["ok"]
    found = svc.find_person_files("mom")
    assert found and len(found[1]) >= 2
    q = svc.person_in_query("show me photos of Mom at the beach")
    assert q and q[0]["name"] == "Mom" and q[1] == "beach"
    assert svc.person_in_query("momentum") is None        # whole words only
    assert svc.person_in_query("photos of Person 2") is None  # unnamed never match

    counts = svc.forget_all()
    assert counts["faces"] == 5 and svc.list_people() == [] and db.count_unscanned_images() == 4


def test_naming_to_existing_name_merges(tmp_path):
    db = make_db(tmp_path)
    for n in ("a", "b", "c", "d"):
        add_file(db, f"/p/{n}.jpg")
    svc = face_service(db, {"a.jpg": [[1, 0]], "b.jpg": [[1, 0]], "c.jpg": [[0, 1]], "d.jpg": [[0, 1]]})
    svc.scan()
    p1, p2 = svc.list_people()
    svc.name_person(p1["id"], "Sam")
    out = svc.name_person(p2["id"], "sam")
    assert out["merged"] and len(svc.list_people()) == 1 and svc.list_people()[0]["photo_count"] == 4


def test_scan_records_per_photo_failure_and_continues(tmp_path):
    db = make_db(tmp_path)
    add_file(db, "/p/a.jpg")
    add_file(db, "/p/b.jpg")
    out = face_service(db, {"a.jpg": ValueError("corrupt"), "b.jpg": [[1, 0]]}).scan()
    assert out["failed"] == 1 and out["scanned"] == 1


def test_new_faces_join_existing_named_person(tmp_path):
    db = make_db(tmp_path)
    for n in ("a", "b"):
        add_file(db, f"/p/{n}.jpg")
    svc = face_service(db, {"a.jpg": [[1, 0]], "b.jpg": [[1, 0]], "c.jpg": [[.99, .05]]})
    svc.scan()
    svc.name_person(svc.list_people()[0]["id"], "Dad")
    add_file(db, "/p/c.jpg")
    out = svc.scan()
    assert out["added_to_existing"] == 1 and out["new_people"] == 0


def test_engine_search_routes_known_person(tmp_path):
    db = make_db(tmp_path)
    e = make_engine(tmp_path)
    e.db = db
    for n in ("a", "b"):
        fid = add_file(db, f"/p/{n}.jpg")
        db.update_file_analysis(fid, "at the beach" if n == "a" else "indoors", "[]", None, "x", False, None)
    e.faces = face_service(db, {"a.jpg": [[1, 0]], "b.jpg": [[1, 0]]})
    e.faces.scan()
    e.faces.name_person(e.faces.list_people()[0]["id"], "Mom")
    r = e.handle("search", {"raw_query": "photos of Mom at the beach"}, {})
    assert "Found 1 photo(s) of Mom matching 'beach'" in r["response"]
    r = e.handle("search", {"raw_query": "photos of Mom on the moon"}, {})
    assert "none matched" in r["response"]


def test_engine_face_intents_when_disabled(tmp_path):
    e = make_engine(tmp_path)
    for intent in ("scan_faces", "list_people", "name_person", "iris_find_person"):
        r = e.handle(intent, {"person": "1", "name": "X"}, {})
        assert "switched off" in r["response"]


# ---------------------------------------------------------------------------
# #77 — detection
# ---------------------------------------------------------------------------

class SeqDetector:
    unavailable_reason = None

    def __init__(self, seq):
        self.seq = iter(seq)

    def detect(self, frame):
        return [detection.Detection(l, c) for l, c in next(self.seq)]


def test_analyse_stream_events_and_debounce():
    seq = [[("laptop", .9)], [("laptop", .9), ("cup", .8)], [("laptop", .9), ("cup", .8)],
           [("laptop", .9)], [("laptop", .9)], [("laptop", .9)]]
    r = detection.analyse_stream(SeqDetector(seq), [(i * 0.5, None) for i in range(6)])
    assert r.events == [(0.0, "appeared", "laptop"), (0.5, "appeared", "cup"), (1.5, "disappeared", "cup")]
    assert r.max_count == {"laptop": 1, "cup": 1}


def test_analyse_stream_ignores_single_frame_flicker():
    seq = [[("cup", .9)], [], [], []]
    r = detection.analyse_stream(SeqDetector(seq), [(i, None) for i in range(4)])
    assert r.events == [] and r.max_count == {"cup": 1}


def test_analyse_stream_confidence_filter_and_counts():
    seq = [[("cup", .9), ("cup", .8), ("tv", .1)]]
    r = detection.analyse_stream(SeqDetector(seq), [(0.0, None)], min_confidence=0.4)
    assert r.max_count == {"cup": 2}


def test_analyse_stream_detector_error_keeps_partial():
    class D:
        n = 0

        def detect(self, f):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("boom")
            return [detection.Detection("cup", .9)]
    r = detection.analyse_stream(D(), [(0, None), (1, None), (2, None)])
    assert r.frames == 1 and r.error == "boom"
    assert "Stopped early" in detection.summarise(r)


def test_summarise_variants():
    assert "didn't get any frames" in detection.summarise(detection.WatchResult())
    assert "I can see: cup ×2" in detection.summarise(
        detection.WatchResult(frames=1, max_count={"cup": 2}), watched=False)
    assert "components or gestures" in detection.summarise(detection.WatchResult(frames=1))


def test_camera_frames_bounded_and_skips_dead_frames():
    t = [0.0]

    class Cam:
        n = 0

        def read(self):
            self.n += 1
            return None if self.n == 2 else "frame"
    out = list(detection.camera_frames(Cam(), 2.0, 0.5, clock=lambda: t[0],
                                       sleep=lambda s: t.__setitem__(0, t[0] + s)))
    assert [round(x[0], 1) for x in out] == [0.0, 1.0, 1.5, 2.0][:len(out)] and len(out) == 4
    # the requested length is capped
    t[0] = 0.0
    capped = list(detection.camera_frames(type("C", (), {"read": lambda s: "f"})(), 10_000, 30.0,
                                          clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s)))
    assert len(capped) == 3  # t=0, 30, 60


def test_objects_json_and_db_search_is_exact_key(tmp_path):
    db = make_db(tmp_path)
    a, b = add_file(db, "/p/a.jpg"), add_file(db, "/p/b.jpg")
    db.update_file_objects(a, detection.objects_json([detection.Detection("cup", .9)] * 2))
    db.update_file_objects(b, json.dumps({"cupboard": 1}))
    assert [r["id"] for r in db.search_files_by_objects("cup")] == [a]
    assert db.search_files_by_objects("") == []


def test_detect_objects_camera_off_by_default(tmp_path):
    e = make_engine(tmp_path, detector=SeqDetector([]))
    assert "switched off" in e.detect_objects()["response"]


def test_detect_objects_detector_missing(tmp_path):
    class Dead:
        unavailable_reason = "ultralytics isn't installed"
    e = make_engine(tmp_path, detector=Dead(), camera=True)
    assert "ultralytics" in e.detect_objects()["response"]


def test_detect_objects_camera_released_even_on_error(tmp_path):
    state = {"closed": False}

    class Cam:
        def __enter__(self):
            return self

        def read(self):
            return "frame"

        def __exit__(self, *a):
            state["closed"] = True
    e = make_engine(tmp_path, detector=SeqDetector([[("cup", .9)]]), camera=True, camera_factory=Cam)
    r = e.detect_objects()
    assert state["closed"] and "cup" in r["response"]


def test_detect_objects_camera_unavailable(tmp_path):
    class Cam:
        def __enter__(self):
            raise detection.CameraUnavailable("couldn't open camera 0")

        def __exit__(self, *a):
            pass
    e = make_engine(tmp_path, detector=SeqDetector([]), camera=True, camera_factory=Cam)
    assert "couldn't open camera" in e.detect_objects()["response"]


def test_detect_objects_on_photo_saves_and_makes_searchable(tmp_path):
    p = tmp_path / "desk.jpg"
    p.write_bytes(b"x")
    e = make_engine(tmp_path, detector=SeqDetector([[("laptop", .9), ("cup", .2)]]))
    fid = add_file(e.db, p)
    r = e.detect_objects(file_id=fid)
    assert "laptop" in r["response"] and "cup" not in r["response"]
    assert json.loads(e.db.get_file(fid)["objects"]) == {"laptop": 1}


def test_can_handle_new_intents(tmp_path):
    e = make_engine(tmp_path)
    for i in ("iris_reindex", "find_similar", "iris_scan_faces", "list_people", "iris_name_person",
              "find_person", "iris_forget_faces", "detect_objects"):
        assert e.can_handle(i)
