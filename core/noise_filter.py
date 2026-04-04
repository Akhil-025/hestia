# core/noise_filter.py

import sys
import numpy as np
import noisereduce as nr
from typing import Optional

class NoiseFilter:
    """
    Stationary noise reduction for int16 PCM audio normalized to float32 [-1.0, 1.0].
    Uses noisereduce.reduce_noise with stationary=True.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the noise filter.

        Args:
            enabled: If False, filter() becomes a passthrough.
        """
        self._enabled = enabled

    @property
    def is_enabled(self) -> bool:
        """Read-only property indicating whether filtering is active."""
        return self._enabled

    def enable(self) -> None:
        """Enable noise reduction."""
        self._enabled = True

    def disable(self) -> None:
        """Disable noise reduction (passthrough mode)."""
        self._enabled = False

    def filter(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Apply stationary noise reduction to the audio array.

        Args:
            audio: Input audio as float32 numpy array, values in [-1.0, 1.0].
            sample_rate: Sample rate in Hz (default 16000).

        Returns:
            Filtered audio (same shape and dtype). If disabled or on error,
            returns the original audio unchanged.

        Raises:
            No exceptions; errors are caught, logged to stderr, and original audio returned.
        """
        if not self._enabled:
            return audio

        try:
            filtered = nr.reduce_noise(y=audio, sr=sample_rate, stationary=True)
            # Ensure output is float32 (noisereduce returns float64 sometimes)
            return filtered.astype(np.float32)
        except Exception as e:
            print(f"NoiseFilter warning: {e}", file=sys.stderr)
            return audio


if __name__ == "__main__":
    # Smoke test
    nf = NoiseFilter(enabled=True)

    # Generate 1 second of white noise at 16kHz
    duration = 1.0
    sr = 16000
    noise = np.random.randn(int(sr * duration)).astype(np.float32)

    # Test with enabled=True
    filtered = nf.filter(noise, sr)
    assert filtered.shape == noise.shape, "Shape mismatch"
    assert filtered.dtype == np.float32, "Output dtype is not float32"

    # Test with enabled=False
    nf.disable()
    passthrough = nf.filter(noise, sr)
    assert np.array_equal(passthrough, noise), "Passthrough changed the audio"

    # Re-enable and test property
    nf.enable()
    assert nf.is_enabled is True
    print("All smoke tests passed.")