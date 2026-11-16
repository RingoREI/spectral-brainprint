"""Frozen V1 Welch and integration conventions; migrated without numeric edits."""
import numpy as np
from scipy import signal

def welch_psd(eeg, fs=300, nperseg=600):
    """Input (..., time, channel), output (..., frequency, channel)."""
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.shape[-2:] != (1200, 18) or not np.isfinite(eeg).all():
        raise ValueError(f"Expected finite (..., 1200, 18) EEG, got {eeg.shape}")
    window = signal.get_window("hann", nperseg, fftbins=True)
    return signal.welch(eeg, fs=fs, window=window, nperseg=nperseg,
                        noverlap=nperseg // 2, nfft=nperseg, detrend="constant",
                        return_onesided=True, scaling="density", axis=-2, average="mean")


def band_power(psd, frequencies, low, high):
    """Trapezoidal integration on an explicit, boundary-aligned grid."""
    mask = (frequencies >= low) & (frequencies <= high)
    selected = frequencies[mask]
    if len(selected) < 2 or not np.isclose(selected[0], low) or not np.isclose(selected[-1], high):
        raise ValueError("Band boundaries must align with the frequency grid")
    return np.trapezoid(psd[..., mask, :], x=selected, axis=-2)


