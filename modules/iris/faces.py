"""
modules/iris/faces.py  (backlog #72: face clustering, local-only)

"Photos of Mom" without any external API. Faces are found and described by
two small models run on this machine through OpenCV; the numbers that come
out (a 128-value "embedding" per face) are grouped by similarity into
unnamed "people", and you give a group a name. Nothing is sent anywhere.

Privacy, because a face embedding is biometric data
---------------------------------------------------
* **Off by default.** ``iris.faces.enabled`` must be true. Until then Iris
  never reads a face.
* **Stored only in Iris's own SQLite file** (tables ``faces``, ``people``,
  ``face_scans``). No face image is saved; only a box and the embedding.
* **Never named automatically.** Groups appear as "Person 3" until *you* name
  one. Naming a group the same as an existing person merges them, which is
  also how you fix a person split across two groups.
* **One command removes it all** (``iris_forget_faces``).
* Hestia's other modules never see this data.

How it works
------------
OpenCV's YuNet finds faces and SFace turns each into an embedding; two ONNX
model files are needed (``face_detection_yunet_*.onnx`` and
``face_recognition_sface_*.onnx``, from the opencv_zoo project), set in
``iris.faces.detector_model`` / ``recognizer_model``. They are not bundled.
Without them, or without OpenCV 4.5.4+, face grouping reports itself
unavailable and nothing else in Iris is affected. The backend is a small
interface (``extract(path) -> [FaceDetection]``) so another model can be
substituted without touching the grouping code.

Grouping is deliberately conservative. Wrongly merging two people is worse
than leaving one person as two groups (which you can merge in one command),
so the default similarity threshold is stricter than SFace's published
same-person cutoff (0.363). Faces below ``face_min_size`` pixels are ignored
because tiny faces produce unreliable embeddings. Accuracy on children,
profiles, masks and low light is poor, as with any such model.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 60
_MAX_SIDE = 1600          # larger photos are shrunk before detection, for speed
_FILLER = frozenset({
    "a", "an", "the", "of", "with", "and", "in", "on", "at", "me", "my", "show",
    "find", "get", "give", "all", "any", "photo", "photos", "picture", "pictures",
    "pic", "pics", "image", "images", "that", "have", "has", "featuring", "contain",
    "containing", "where", "is", "are", "was", "were", "there", "from", "do", "i",
    "you", "please", "look", "for", "up", "us",
})


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def normalise(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else [0.0] * len(vec)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    dim = len(vectors[0])
    return normalise([sum(v[i] for v in vectors) / len(vectors) for i in range(dim)])


# ---------------------------------------------------------------------------
# Grouping (pure: no database, no models)
# ---------------------------------------------------------------------------

def group_faces(
    unassigned: Sequence[dict],
    existing: dict[int, list[float]],
    threshold: float,
    min_size: int = 2,
) -> dict[str, Any]:
    """Decide where each not-yet-grouped face belongs.

    *unassigned*: ``{"id", "embedding", "score"}`` dicts.
    *existing*: ``{person_id: centroid}`` for people already in the database.

    A face joins the existing person it is most similar to if that similarity
    reaches *threshold* (one match is enough, since the person is already
    established). Remaining faces are grouped among themselves, and a new
    group is only kept if it has at least *min_size* faces, so a one-off face
    in the background of a photo doesn't become a "person".

    Faces are visited best-detection-first (ties by id) so the result is the
    same every run and group seeds are the clearest faces.

    Returns ``{"existing": {face_id: person_id}, "new": [[face_id, ...], ...],
    "unclustered": [face_id, ...]}``.
    """
    ordered = sorted(unassigned, key=lambda f: (-float(f.get("score") or 0.0), f["id"]))
    to_existing: dict[int, int] = {}
    groups: list[dict[str, Any]] = []

    for face in ordered:
        vec = normalise(face["embedding"])
        best_pid, best_sim = None, threshold
        for pid, centroid in existing.items():
            sim = cosine(vec, centroid)
            if sim >= best_sim:
                best_pid, best_sim = pid, sim
        if best_pid is not None:
            to_existing[face["id"]] = best_pid
            continue

        best_group, best_sim = None, threshold
        for g in groups:
            sim = cosine(vec, g["centroid"])
            if sim >= best_sim:
                best_group, best_sim = g, sim
        if best_group is None:
            groups.append({"ids": [face["id"]], "vecs": [vec], "centroid": vec})
        else:
            best_group["ids"].append(face["id"])
            best_group["vecs"].append(vec)
            best_group["centroid"] = _centroid(best_group["vecs"])

    # Groups seeded early can drift together as members are added; fold any
    # pair whose centroids now clear the threshold.
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if cosine(groups[i]["centroid"], groups[j]["centroid"]) >= threshold:
                    groups[i]["ids"] += groups[j]["ids"]
                    groups[i]["vecs"] += groups[j]["vecs"]
                    groups[i]["centroid"] = _centroid(groups[i]["vecs"])
                    del groups[j]
                    merged = True
                    break
            if merged:
                break

    kept = [sorted(g["ids"]) for g in groups if len(g["ids"]) >= max(1, min_size)]
    kept.sort(key=lambda ids: (-len(ids), ids[0]))
    clustered = {i for ids in kept for i in ids}
    return {
        "existing": to_existing,
        "new": kept,
        "unclustered": sorted(f["id"] for f in ordered
                              if f["id"] not in to_existing and f["id"] not in clustered),
    }


# ---------------------------------------------------------------------------
# Backend: OpenCV YuNet + SFace
# ---------------------------------------------------------------------------

@dataclass
class FaceDetection:
    box: tuple          # (x, y, w, h) as fractions of the image, 0-1
    score: float
    embedding: list


def _load_cv2():
    try:
        import cv2
        return cv2
    except Exception as exc:
        logger.warning("[Iris] OpenCV unavailable (%s); face grouping is off.", exc)
        return None


class OpenCVFaceBackend:
    """Finds and embeds faces with OpenCV's YuNet and SFace models."""

    def __init__(
        self,
        detector_model: str = "",
        recognizer_model: str = "",
        score_threshold: float = 0.85,
        min_face_size: int = 40,
    ) -> None:
        self.detector_model = detector_model
        self.recognizer_model = recognizer_model
        self.score_threshold = score_threshold
        self.min_face_size = min_face_size
        self._cv2 = None
        self._detector = None
        self._recognizer = None
        self._failed: Optional[str] = None

    @property
    def unavailable_reason(self) -> Optional[str]:
        """Why face grouping can't run, or None if it can. Loads the models."""
        self._ensure_loaded()
        return self._failed

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def _ensure_loaded(self) -> None:
        if self._detector is not None or self._failed:
            return
        if not self.detector_model or not self.recognizer_model:
            self._failed = ("no face models configured (set iris.faces.detector_model "
                            "and iris.faces.recognizer_model)")
            return
        for label, p in (("detector", self.detector_model), ("recognizer", self.recognizer_model)):
            if not Path(p).is_file():
                self._failed = f"face {label} model not found at {p}"
                return
        cv2 = self._cv2 or _load_cv2()
        if cv2 is None:
            self._failed = "OpenCV is not installed"
            return
        if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
            self._failed = "this OpenCV is too old for face models (need 4.5.4 or newer)"
            return
        try:
            self._detector = cv2.FaceDetectorYN.create(
                self.detector_model, "", (320, 320), self.score_threshold)
            self._recognizer = cv2.FaceRecognizerSF.create(self.recognizer_model, "")
            self._cv2 = cv2
        except Exception as exc:
            self._detector = None
            self._failed = f"could not load the face models: {exc}"

    def extract(self, path: "str | Path") -> list[FaceDetection]:
        """Every usable face in the image. Raises if the image can't be read
        or the backend is unavailable (the caller records that per file)."""
        reason = self.unavailable_reason
        if reason:
            raise RuntimeError(reason)
        cv2 = self._cv2
        import numpy as np
        data = np.fromfile(str(path), dtype=np.uint8)   # unicode-safe on Windows
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            raise ValueError(f"couldn't decode image {path}")
        h, w = img.shape[:2]
        scale = min(1.0, _MAX_SIDE / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]

        self._detector.setInputSize((w, h))
        _status, rows = self._detector.detect(img)
        out: list[FaceDetection] = []
        if rows is None:
            return out
        for row in rows:
            x, y, fw, fh = (float(v) for v in row[:4])
            if min(fw, fh) < self.min_face_size:
                continue
            aligned = self._recognizer.alignCrop(img, row)
            feat = self._recognizer.feature(aligned)
            emb = normalise([float(v) for v in np.asarray(feat).reshape(-1)])
            out.append(FaceDetection(
                box=(round(x / w, 4), round(y / h, 4), round(fw / w, 4), round(fh / h, 4)),
                score=float(row[-1]),
                embedding=emb,
            ))
        return out


# ---------------------------------------------------------------------------
# Service: scanning, grouping, naming, finding
# ---------------------------------------------------------------------------

def clean_name(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    name = re.sub(r"\s+", " ", name).strip().strip(".,!?\"'")
    if not (1 <= len(name) <= MAX_NAME_LENGTH):
        return None
    return name


class FaceService:
    """Face scanning and people management over an IrisDB."""

    def __init__(
        self,
        db,
        backend=None,
        *,
        enabled: bool = False,
        threshold: float = 0.45,
        min_cluster_size: int = 2,
    ) -> None:
        self.db = db
        self.backend = backend
        self.enabled = enabled
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size

    # -- availability --------------------------------------------------

    def unavailable_reason(self) -> Optional[str]:
        if not self.enabled:
            return ("Face grouping is switched off. Turn on iris.faces.enabled in "
                    "your config if you want it; faces are only ever processed on this machine.")
        if self.backend is None:
            return "No face model is set up."
        reason = getattr(self.backend, "unavailable_reason", None)
        return f"Face grouping isn't available: {reason}." if reason else None

    # -- scanning and grouping ----------------------------------------

    def scan(self, limit: int = 100) -> dict:
        """Find faces in photos not yet scanned, then group the new faces.
        Never raises; per-photo failures are counted."""
        blocked = self.unavailable_reason()
        if blocked:
            return {"available": False, "message": blocked}
        limit = max(1, min(int(limit or 100), 2000))
        pending = self.db.get_unscanned_images(limit)
        scanned = failed = faces_found = 0
        for row in pending:
            try:
                found = self.backend.extract(row["file_path"])
            except Exception as exc:
                logger.warning("[Iris] Face scan failed for %s: %s", row["file_path"], exc)
                self.db.record_face_scan(row["id"], 0, error=str(exc)[:200])
                failed += 1
                continue
            for f in found:
                self.db.add_face(row["id"], json.dumps(list(f.box)), f.score,
                                 json.dumps([round(x, 6) for x in f.embedding]))
            self.db.record_face_scan(row["id"], len(found))
            scanned += 1
            faces_found += len(found)
        grouping = self.group_new_faces()
        return {
            "available": True, "scanned": scanned, "failed": failed,
            "faces_found": faces_found, "remaining": max(0, self.db.count_unscanned_images()),
            **grouping,
        }

    def _centroids(self) -> dict[int, list[float]]:
        by_person: dict[int, list[list[float]]] = {}
        for f in self.db.get_faces():
            if f["person_id"] is not None:
                by_person.setdefault(f["person_id"], []).append(normalise(_load_vec(f)))
        return {pid: _centroid(vs) for pid, vs in by_person.items() if vs}

    def group_new_faces(self) -> dict:
        loose = [{"id": f["id"], "embedding": _load_vec(f), "score": f["score"]}
                 for f in self.db.get_faces(unassigned_only=True)]
        loose = [f for f in loose if f["embedding"]]
        if not loose:
            return {"new_people": 0, "added_to_existing": 0, "unclustered": 0}
        plan = group_faces(loose, self._centroids(), self.threshold, self.min_cluster_size)
        for face_id, pid in plan["existing"].items():
            self.db.set_face_person([face_id], pid)
        for ids in plan["new"]:
            self.db.set_face_person(ids, self.db.create_person())
        return {
            "new_people": len(plan["new"]),
            "added_to_existing": len(plan["existing"]),
            "unclustered": len(plan["unclustered"]),
        }

    # -- people ---------------------------------------------------------

    @staticmethod
    def label(person: dict) -> str:
        return person.get("name") or f"Person {person['id']}"

    def list_people(self) -> list[dict]:
        people = self.db.get_people()
        for p in people:
            p["label"] = self.label(p)
        return people

    def resolve(self, ref: Any) -> Optional[dict]:
        """A person by id, by "Person 3", or by name (case-insensitive)."""
        if ref is None:
            return None
        people = self.list_people()
        text = str(ref).strip()
        m = re.fullmatch(r"(?:person\s*)?#?(\d+)", text, re.IGNORECASE)
        if m:
            wanted = int(m.group(1))
            for p in people:
                if p["id"] == wanted:
                    return p
        low = text.lower()
        for p in people:
            if (p.get("name") or "").lower() == low:
                return p
        return None

    def name_person(self, ref: Any, name: Any) -> dict:
        person = self.resolve(ref)
        if person is None:
            return {"ok": False, "reason": "I couldn't find that person."}
        new_name = clean_name(name)
        if new_name is None:
            return {"ok": False, "reason": f"A name needs to be 1–{MAX_NAME_LENGTH} characters."}
        other = self.resolve(new_name)
        if other is not None and other["id"] != person["id"]:
            self.db.merge_people(person["id"], other["id"])
            return {"ok": True, "merged": True, "person_id": other["id"], "name": other.get("name") or new_name,
                    "label": self.label(other)}
        self.db.rename_person(person["id"], new_name)
        return {"ok": True, "merged": False, "person_id": person["id"], "name": new_name, "label": new_name}

    def find_person_files(self, ref: Any) -> Optional[tuple[dict, list[dict]]]:
        person = self.resolve(ref)
        if person is None:
            return None
        return person, self.db.get_files_for_person(person["id"])

    def person_in_query(self, query: str) -> Optional[tuple[dict, str]]:
        """If *query* names someone Iris knows, ``(person, remainder)`` where
        remainder is what's left after the name and filler words ("photos of
        Mom at the beach" -> "beach"). Only named people match, never "Person
        3", and only whole words, longest name first."""
        if not self.enabled or not query:
            return None
        named = sorted((p for p in self.list_people() if p.get("name")),
                       key=lambda p: -len(p["name"]))
        low = query.lower()
        for p in named:
            pat = r"(?<!\w)" + re.escape(p["name"].lower()) + r"(?!\w)"
            if len(p["name"]) >= 2 and re.search(pat, low):
                rest = re.sub(pat, " ", low)
                words = [w for w in re.findall(r"[a-z0-9']+", rest) if w not in _FILLER]
                return p, " ".join(words)
        return None

    def forget_all(self) -> dict:
        """Delete every face, person and scan record."""
        return self.db.delete_all_face_data()


def _load_vec(face_row: dict) -> list[float]:
    try:
        v = json.loads(face_row["embedding"])
        return [float(x) for x in v] if isinstance(v, list) else []
    except (TypeError, ValueError, KeyError):
        return []
