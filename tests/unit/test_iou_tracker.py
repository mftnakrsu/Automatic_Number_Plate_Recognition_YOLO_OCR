from __future__ import annotations

import pytest

from anpr.detector.base import BBox, Detection
from anpr.tracker.iou import IoUTracker, iou


def _det(x1: int, y1: int, x2: int, y2: int) -> Detection:
    return Detection(BBox(x1, y1, x2, y2), confidence=0.9)


def test_iou_identical_boxes_is_one() -> None:
    a = BBox(0, 0, 100, 100)
    assert iou(a, a) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    a = BBox(0, 0, 100, 100)
    b = BBox(200, 200, 300, 300)
    assert iou(a, b) == 0.0


def test_tracker_assigns_new_id_to_first_detection() -> None:
    tr = IoUTracker()
    out = tr.update([_det(0, 0, 100, 100)])
    assert len(out) == 1
    assert out[0][0] == 1


def test_tracker_keeps_id_for_overlapping_box() -> None:
    tr = IoUTracker(iou_threshold=0.3)
    first = tr.update([_det(0, 0, 100, 100)])
    second = tr.update([_det(5, 5, 95, 95)])
    assert first[0][0] == second[0][0]


def test_tracker_assigns_new_id_to_disjoint_box() -> None:
    tr = IoUTracker(iou_threshold=0.3)
    a = tr.update([_det(0, 0, 100, 100)])
    b = tr.update([_det(500, 500, 600, 600)])
    assert a[0][0] != b[0][0]


def test_tracker_expires_stale_track() -> None:
    tr = IoUTracker(max_age=2)
    tr.update([_det(0, 0, 100, 100)])
    tr.update([])
    tr.update([])
    tr.update([])
    assert tr.active_track_count == 0


def test_tracker_iou_threshold_validation() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        IoUTracker(iou_threshold=2.0)
