"""
modules/iris/detection.py  (backlog #77: live object detection, scoped down)

What this is: "what's on my desk?" and "watch the camera for 20 seconds and
tell me what comes and goes", using a small YOLO model over the 80 everyday
object classes it was trained on (person, laptop, cup, phone, keyboard,
scissors, ...). It can also run on a saved photo to fill that photo's
``objects`` field, which makes "photos with a laptop" searchable.

What it is not: the PCB-fault and gesture recognition HEARTH.txt imagines.
YOLO's stock classes don't include components, solder bridges or hand
gestures, so this will not diagnose hardware or read gestures. It says what
common objects are in view, and when they appear or disappear.

Privacy
-------
* **The camera is off unless you enable it** (``iris.camera.enabled``). The
  camera is opened only for the length of one request and always released.
* **Frames are held in memory, analysed and dropped.** Nothing is written to
  disk unless you ask for a photo to be saved.
* Everything runs locally; the only network use is the one-time download of
  the model weights by ``ultralytics`` the first time the model is used.

Needs ``ultralytics`` (YOLO) and OpenCV. Either missing is reported plainly
and the rest of Iris is unaffected.

Standalone, for a live view in a terminal::

    python -m modules.iris.detection --seconds 30
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

MAX_WATCH_SECONDS = 60.0
DEFAULT_WATCH_SECONDS = 10.0
DEFAULT_INTERVAL = 0.5
STABLE_FRAMES = 2


class CameraUnavailable(RuntimeError):
    """The camera couldn't be opened, or OpenCV isn't installed."""


class DetectorUnavailable(RuntimeError):
    """The detection model couldn't be loaded."""


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple = (0.0, 0.0, 0.0, 0.0)    # x1, y1, x2, y2 as fractions of the frame


class YoloDetector:
    """Lazily loaded YOLO model. Constructing it costs nothing."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.4) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self._model = None
        self._failed: Optional[str] = None

    @property
    def unavailable_reason(self) -> Optional[str]:
        self._ensure_loaded()
        return self._failed

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._failed:
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self._failed = f"ultralytics isn't installed ({type(exc).__name__})"
            return
        try:
            self._model = YOLO(self.model_name)
        except Exception as exc:
            self._failed = f"couldn't load model {self.model_name!r}: {exc}"

    def detect(self, image: Any) -> list[Detection]:
        """Objects in *image* (a file path or an OpenCV frame)."""
        reason = self.unavailable_reason
        if reason:
            raise DetectorUnavailable(reason)
        results = self._model.predict(image, conf=self.confidence, verbose=False)
        if not results:
            return []
        r = results[0]
        names = getattr(r, "names", {}) or {}
        out: list[Detection] = []
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return out
        for b in boxes:
            cls = int(b.cls[0])
            out.append(Detection(
                label=str(names.get(cls, cls)),
                confidence=float(b.conf[0]),
                box=tuple(float(v) for v in b.xyxyn[0].tolist()),
            ))
        return out


class Camera:
    """A webcam, opened for the length of a ``with`` block and always released."""

    def __init__(self, index: int = 0, cv2_module: Any = None) -> None:
        self.index = index
        self._cv2 = cv2_module
        self._cap = None

    def __enter__(self) -> "Camera":
        cv2 = self._cv2
        if cv2 is None:
            try:
                import cv2
            except Exception as exc:
                raise CameraUnavailable(f"OpenCV isn't installed ({type(exc).__name__})") from exc
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise CameraUnavailable(f"couldn't open camera {self.index} (in use, unplugged or blocked)")
        self._cap = cap
        return self

    def read(self) -> Any:
        ok, frame = self._cap.read()
        return frame if ok else None

    def __exit__(self, *exc: Any) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def camera_frames(
    camera: Any,
    seconds: float,
    interval: float = DEFAULT_INTERVAL,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[tuple[float, Any]]:
    """``(seconds_since_start, frame)`` roughly every *interval* seconds for
    *seconds* seconds, always including one frame at t=0. Frames the camera
    fails to deliver are skipped."""
    seconds = max(0.0, min(float(seconds), MAX_WATCH_SECONDS))
    interval = max(0.05, float(interval))
    start = clock()
    while True:
        t = clock() - start
        frame = camera.read()
        if frame is not None:
            yield t, frame
        if t + interval > seconds:
            return
        sleep(interval)


@dataclass
class WatchResult:
    frames: int = 0
    seconds: float = 0.0
    max_count: dict = field(default_factory=dict)     # label -> most seen at once
    frames_seen: dict = field(default_factory=dict)   # label -> frames it was in
    events: list = field(default_factory=list)        # (t, "appeared"|"disappeared", label)
    error: Optional[str] = None


def analyse_stream(
    detector: Any,
    frames: Iterable[tuple[float, Any]],
    min_confidence: float = 0.4,
    stable_frames: int = STABLE_FRAMES,
) -> WatchResult:
    """Run *detector* over *frames* and report what was seen and what changed.

    A label only counts as having appeared once it is in ``stable_frames``
    consecutive frames, and as gone once it is missing from that many in a
    row, so one flickering detection doesn't generate events. The time given
    for an event is that of the first frame of the streak. Stops at the first
    detector failure and reports it in ``error`` with what was gathered so far.
    """
    res = WatchResult()
    present: dict[str, bool] = {}
    streak: dict[str, int] = {}
    streak_start: dict[str, float] = {}
    last_t = 0.0
    for t, frame in frames:
        try:
            dets = [d for d in detector.detect(frame) if d.confidence >= min_confidence]
        except Exception as exc:
            res.error = str(exc)
            break
        res.frames += 1
        last_t = t
        counts: dict[str, int] = {}
        for d in dets:
            counts[d.label] = counts.get(d.label, 0) + 1
        for label, n in counts.items():
            res.max_count[label] = max(res.max_count.get(label, 0), n)
            res.frames_seen[label] = res.frames_seen.get(label, 0) + 1
        for label in set(present) | set(counts):
            here = label in counts
            was = present.get(label, False)
            if here == was:
                streak[label] = 0
                continue
            if streak.get(label, 0) == 0:
                streak_start[label] = t
            streak[label] = streak.get(label, 0) + 1
            if streak[label] >= max(1, stable_frames):
                present[label] = here
                streak[label] = 0
                res.events.append((round(streak_start[label], 1),
                                   "appeared" if here else "disappeared", label))
    res.seconds = round(last_t, 1)
    res.events.sort(key=lambda e: e[0])
    return res


def summarise(result: WatchResult, watched: bool = True) -> str:
    """Plain-language description of a WatchResult."""
    if result.error and result.frames == 0:
        return f"I couldn't run detection: {result.error}."
    if result.frames == 0:
        return "I didn't get any frames from the camera."
    if not result.max_count:
        text = ("I don't see any objects I recognise." if not watched else
                f"Over {result.seconds:g}s ({result.frames} frames) I didn't see any objects I recognise.")
    else:
        parts = [f"{label} ×{n}" if n > 1 else label
                 for label, n in sorted(result.max_count.items(), key=lambda kv: (-kv[1], kv[0]))]
        if watched:
            text = f"Over {result.seconds:g}s ({result.frames} frames) I saw: " + ", ".join(parts) + "."
        else:
            text = "I can see: " + ", ".join(parts) + "."
    if watched and result.events:
        shown = [f"{label} {what} at {t:g}s" for t, what, label in result.events[:8]]
        text += " Changes: " + "; ".join(shown) + "."
    if result.error:
        text += f" (Stopped early: {result.error}.)"
    text += " I only recognise everyday objects, not components or gestures."
    return text


def objects_json(detections: Iterable[Detection]) -> str:
    """The value stored in the ``objects`` column: {"laptop": 1, "cup": 2}."""
    import json
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    return json.dumps(dict(sorted(counts.items())))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Live object detection from the camera.")
    p.add_argument("--seconds", type=float, default=DEFAULT_WATCH_SECONDS)
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--confidence", type=float, default=0.4)
    args = p.parse_args(argv)

    detector = YoloDetector(args.model, args.confidence)
    if not detector.available:
        print(f"Can't run: {detector.unavailable_reason}")
        return 2
    try:
        with Camera(args.camera) as cam:
            def live() -> Iterator[tuple[float, Any]]:
                for t, frame in camera_frames(cam, args.seconds, args.interval):
                    dets = detector.detect(frame)
                    seen = ", ".join(sorted({d.label for d in dets})) or "nothing recognised"
                    print(f"[{t:5.1f}s] {seen}", flush=True)
                    yield t, frame
            result = analyse_stream(detector, live(), args.confidence)
    except CameraUnavailable as exc:
        print(f"Can't run: {exc}")
        return 2
    print()
    print(summarise(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
