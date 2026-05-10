"""Temporal majority voting per track ID.

For each track, we accumulate top-K confidence-weighted reads. Once the dwell
threshold is met, the confirmed plate is the per-character-position majority
of those reads. The 2025 UFPR-SR-Plates benchmark (arXiv:2505.06393) showed
this single trick lifts accuracy on low-resolution plates from ~31% to ~45%.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(slots=True)
class _TrackHistory:
    reads: deque[tuple[str, float]]
    confirmed: str | None = None


class TemporalVoter:
    def __init__(self, *, window: int = 8, min_dwell: int = 5, top_k: int = 3) -> None:
        if window < min_dwell:
            raise ValueError("window must be >= min_dwell")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._window = window
        self._min_dwell = min_dwell
        self._top_k = top_k
        self._tracks: dict[int, _TrackHistory] = defaultdict(self._new_history)

    def _new_history(self) -> _TrackHistory:
        return _TrackHistory(reads=deque(maxlen=self._window))

    def observe(self, track_id: int, text: str, confidence: float) -> str | None:
        """Record one read. Returns the confirmed plate when dwell ≥ min_dwell."""
        h = self._tracks[track_id]
        h.reads.append((text, confidence))
        if len(h.reads) < self._min_dwell:
            return None
        h.confirmed = _vote(h.reads, top_k=self._top_k)
        return h.confirmed

    def confirmed_for(self, track_id: int) -> str | None:
        h = self._tracks.get(track_id)
        return h.confirmed if h else None

    def forget(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)


def _vote(reads: deque[tuple[str, float]], *, top_k: int) -> str:
    top = sorted(reads, key=lambda r: r[1], reverse=True)[:top_k]
    if not top:
        return ""
    max_len = max(len(t) for t, _ in top)
    chars: list[str] = []
    for i in range(max_len):
        weights: dict[str, float] = defaultdict(float)
        for text, conf in top:
            if i < len(text):
                weights[text[i]] += conf
        if weights:
            chars.append(max(weights.items(), key=lambda kv: kv[1])[0])
    return "".join(chars)
