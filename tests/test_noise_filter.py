# tests/test_noise_filter.py
"""
Regression tests for core/noise_filter.py.

Run with:  pytest tests/test_noise_filter.py -v

Covers the enable/disable/passthrough contract and the "never raises,
falls back to unfiltered audio on error" guarantee the docstring promises.
The actual `noisereduce.reduce_noise` call is mocked so these tests don't
depend on real DSP output — only on NoiseFilter's own control flow.
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.noise_filter import NoiseFilter


def _white_noise(seconds: float = 0.1, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal(int(sr * seconds)).astype(np.float32)


def test_disabled_filter_is_a_passthrough():
    nf = NoiseFilter(enabled=False)
    audio = _white_noise()

    result = nf.filter(audio)

    assert np.array_equal(result, audio)


def test_disabled_filter_never_calls_reduce_noise():
    nf = NoiseFilter(enabled=False)
    with patch("core.noise_filter.nr.reduce_noise") as mock_reduce:
        nf.filter(_white_noise())
    mock_reduce.assert_not_called()


def test_enabled_filter_calls_reduce_noise_and_casts_to_float32():
    nf = NoiseFilter(enabled=True)
    audio = _white_noise()

    with patch("core.noise_filter.nr.reduce_noise") as mock_reduce:
        # noisereduce sometimes returns float64 — filter() must cast back.
        mock_reduce.return_value = audio.astype(np.float64) * 0.5
        result = nf.filter(audio, sample_rate=16000)

    mock_reduce.assert_called_once()
    _, kwargs = mock_reduce.call_args
    assert kwargs["sr"] == 16000
    assert kwargs["stationary"] is True
    assert result.dtype == np.float32


def test_filter_returns_original_audio_on_internal_error():
    nf = NoiseFilter(enabled=True)
    audio = _white_noise()

    with patch("core.noise_filter.nr.reduce_noise", side_effect=RuntimeError("dsp blew up")):
        result = nf.filter(audio)

    # Per the docstring: never raises, falls back to the original audio.
    assert np.array_equal(result, audio)


def test_enable_and_disable_toggle_is_enabled_property():
    nf = NoiseFilter(enabled=False)
    assert nf.is_enabled is False

    nf.enable()
    assert nf.is_enabled is True

    nf.disable()
    assert nf.is_enabled is False


def test_default_construction_is_enabled():
    nf = NoiseFilter()
    assert nf.is_enabled is True
