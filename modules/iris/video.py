"""
modules/iris/video.py  (backlog #75: video support)

Frame sampling for Iris. A video is turned into a handful of still frames,
and from there everything Iris already does for photos applies: each frame
is captioned by the vision model, and the frames' CLIP embeddings are
averaged into ONE embedding for the video, stored under the video's own
file_id. That is deliberate. It means semantic search, "find similar",
album clustering and re-indexing all work on videos without any of them
needing to know that a video is different.

What this does not do
---------------------
* It does not transcribe audio, track motion or read on-screen text. A video
  is described by a few stills taken evenly through it, so something that
  happens in a gap between samples is not seen.
* It needs OpenCV (``opencv-python``). Without it every function here
  returns "nothing" (``None`` / ``[]``) instead of raising, and Iris carries
  on with photos only.
* Container/codec support is whatever your OpenCV build has. If it can't
  open a file, that file is reported as an analysis error and can be retried.
* ``cv2.VideoCapture`` cannot open paths with non-ASCII characters on some
  Windows builds; such files fail the same way.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# A frame whose mean pixel value is below this (0-255) is treated as black
# (a fade-in, a lens cap) and passed over for a brighter one.
BLACK_FRAME_MEAN = 8.0
DEFAULT_FRAMES = 4
MAX_FRAMES = 12


def _load_cv2():
    """OpenCV, or None if it's missing or its native library is broken."""
    try:
        import cv2
        return cv2
    except Exception as exc:  # ImportError, or a DLL/shared-library failure
        logger.warning("[Iris] OpenCV unavailable (%s); video support is off.", exc)
        return None


@dataclass
class VideoInfo:
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


def format_duration(seconds: Optional[float]) -> str:
    """0:42, 12:05 or 1:02:03. Empty string when the length isn't known."""
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def probe_video(path: "str | Path") -> Optional[VideoInfo]:
    """Length and size of a video, or None if it can't be opened."""
    cv2 = _load_cv2()
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if not math.isfinite(fps) or fps <= 0 or frames <= 0:
            duration = 0.0
        else:
            duration = frames / fps
        return VideoInfo(duration, fps if math.isfinite(fps) else 0.0, max(frames, 0), width, height)
    except Exception as exc:
        logger.warning("[Iris] Could not probe %s: %s", path, exc)
        return None
    finally:
        cap.release()


def sample_positions(frame_count: int, n: int) -> list[int]:
    """*n* frame indexes spread evenly through the video, each at the middle of
    its slice (so the first sample isn't the often-black opening frame and the
    last isn't the end card). Fewer than *n* if the video is shorter."""
    if frame_count <= 0 or n <= 0:
        return []
    n = min(n, frame_count)
    return sorted({min(frame_count - 1, int((i + 0.5) * frame_count / n)) for i in range(n)})


def sample_frames(
    path: "str | Path",
    out_dir: "str | Path",
    n: int = DEFAULT_FRAMES,
) -> list[Path]:
    """Write up to *n* evenly spaced frames from *path* as JPEGs in *out_dir*
    and return their paths in order. Never raises.

    Unreadable frames are skipped, and so are near-black ones, unless every
    frame is near-black, in which case the brightest is kept so a dark video
    still yields something. Returns [] if OpenCV is missing, the file can't be
    opened, or its length is unknown.
    """
    cv2 = _load_cv2()
    if cv2 is None:
        return []
    n = max(1, min(int(n or DEFAULT_FRAMES), MAX_FRAMES))
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        positions = sample_positions(total, n)
        if not positions:
            logger.warning("[Iris] %s reports no frame count; can't sample it.", path)
            return []

        grabbed: list = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            grabbed.append((pos, frame, float(frame.mean())))
        if not grabbed:
            return []

        keep = [g for g in grabbed if g[2] >= BLACK_FRAME_MEAN]
        if not keep:
            keep = [max(grabbed, key=lambda g: g[2])]

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for i, (_pos, frame, _mean) in enumerate(keep):
            target = out / f"frame_{i:02d}.jpg"
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            # tofile() rather than cv2.imwrite(): imwrite can't write to a
            # path with non-ASCII characters on Windows.
            buf.tofile(str(target))
            written.append(target)
        return written
    except Exception as exc:
        logger.warning("[Iris] Frame sampling failed for %s: %s", path, exc)
        return []
    finally:
        cap.release()


def mean_embedding(vectors: Sequence[Sequence[float]]) -> Optional[list[float]]:
    """The normalised average of unit-length versions of *vectors*. Averaging
    unit vectors (rather than raw ones) keeps one unusually high-magnitude
    frame from dominating. None for no usable input."""
    usable = []
    for v in vectors or []:
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 0:
            usable.append([x / norm for x in v])
    if not usable:
        return None
    dim = len(usable[0])
    usable = [u for u in usable if len(u) == dim]
    mean = [sum(u[i] for u in usable) / len(usable) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in mean))
    if norm == 0:
        return None
    return [x / norm for x in mean]


def video_embedding(
    embedder,
    frames: Sequence["str | Path"],
) -> Optional[list[float]]:
    """One CLIP embedding for a video: the mean of its frames' embeddings."""
    vectors = []
    for frame in frames:
        try:
            vec = embedder.embed_image(frame)
        except Exception as exc:
            logger.warning("[Iris] Embedding frame %s failed: %s", frame, exc)
            continue
        if vec is not None:
            vectors.append(vec)
    return mean_embedding(vectors)


def combine_frame_results(
    results: Sequence[tuple[Optional[str], Optional[str], Optional[str]]],
    max_captions: int = 4,
    max_tags: int = 10,
) -> tuple[str, list[str], str]:
    """Merge per-frame ``(caption, tags_csv, mood)`` into one record for the
    video: the distinct captions in order, the union of tags (plus ``video``),
    and the most common mood."""
    captions: list[str] = []
    seen_caps: set[str] = set()
    tags: list[str] = ["video"]
    seen_tags = {"video"}
    moods: dict[str, int] = {}
    for caption, tag_csv, mood in results:
        if caption:
            key = caption.strip().lower()
            if key and key not in seen_caps and len(captions) < max_captions:
                seen_caps.add(key)
                captions.append(caption.strip())
        for t in (tag_csv or "").split(","):
            t = t.strip()
            if t and t.lower() not in seen_tags and len(tags) < max_tags:
                seen_tags.add(t.lower())
                tags.append(t)
        if mood:
            m = mood.strip().lower()
            if m:
                moods[m] = moods.get(m, 0) + 1
    caption = "Video: " + " → ".join(captions) if captions else "Unlabeled video"
    mood = max(moods, key=lambda k: (moods[k], -list(moods).index(k))) if moods else "neutral"
    return caption, tags, mood
