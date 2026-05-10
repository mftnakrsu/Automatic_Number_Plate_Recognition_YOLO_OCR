"""Lightweight IoU-greedy tracker for fixed-camera ANPR.

For crowded MOT scenes use a real tracker (ByteTrack / BoT-SORT). For ANPR with
~5-10 plates/frame on a fixed camera at 30fps, IoU-greedy assignment is enough
and dependency-free. Track IDs increment monotonically; stale tracks expire
after `max_age` frames without a match.
"""

from __future__ import annotations

from dataclasses import dataclass

from anpr.detector.base import BBox, Detection


def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass(slots=True)
class _Track:
    track_id: int
    bbox: BBox
    last_frame: int
    history: int = 1


class IoUTracker:
    def __init__(self, *, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._frame_idx = 0

    def update(self, detections: list[Detection]) -> list[tuple[int, Detection]]:
        """Assign track IDs to current frame's detections; expire stale tracks."""
        self._frame_idx += 1
        assigned: list[tuple[int, Detection]] = []
        unmatched: list[int] = list(range(len(detections)))

        for t in list(self._tracks.values()):
            best_iou = 0.0
            best_idx = -1
            for di in unmatched:
                v = iou(t.bbox, detections[di].bbox)
                if v > best_iou:
                    best_iou = v
                    best_idx = di
            if best_iou >= self._iou_threshold and best_idx >= 0:
                d = detections[best_idx]
                t.bbox = d.bbox
                t.last_frame = self._frame_idx
                t.history += 1
                assigned.append((t.track_id, d))
                unmatched.remove(best_idx)

        for di in unmatched:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(
                track_id=tid,
                bbox=detections[di].bbox,
                last_frame=self._frame_idx,
            )
            assigned.append((tid, detections[di]))

        stale = [
            tid for tid, t in self._tracks.items() if self._frame_idx - t.last_frame > self._max_age
        ]
        for tid in stale:
            del self._tracks[tid]

        return assigned

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)
