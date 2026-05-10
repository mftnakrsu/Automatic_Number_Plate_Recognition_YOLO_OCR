from __future__ import annotations

import pytest

from anpr.postprocess.temporal import TemporalVoter


def test_no_emit_before_dwell() -> None:
    voter = TemporalVoter(window=5, min_dwell=3)
    assert voter.observe(1, "34ABC1234", 0.9) is None
    assert voter.observe(1, "34ABC1234", 0.9) is None


def test_emits_after_dwell() -> None:
    voter = TemporalVoter(window=5, min_dwell=3)
    for _ in range(2):
        voter.observe(1, "34ABC1234", 0.9)
    confirmed = voter.observe(1, "34ABC1234", 0.9)
    assert confirmed == "34ABC1234"


def test_majority_wins_per_position() -> None:
    voter = TemporalVoter(window=5, min_dwell=3, top_k=3)
    voter.observe(1, "34ABC1234", 0.95)
    voter.observe(1, "34ABC1234", 0.92)
    confirmed = voter.observe(1, "34ABC1235", 0.50)  # last char differs, low conf
    assert confirmed == "34ABC1234"


def test_window_smaller_than_dwell_rejected() -> None:
    with pytest.raises(ValueError, match="min_dwell"):
        TemporalVoter(window=3, min_dwell=5)


def test_forget_clears_track() -> None:
    voter = TemporalVoter(window=5, min_dwell=2)
    voter.observe(1, "34ABC1234", 0.9)
    voter.observe(1, "34ABC1234", 0.9)
    assert voter.confirmed_for(1) == "34ABC1234"
    voter.forget(1)
    assert voter.confirmed_for(1) is None
