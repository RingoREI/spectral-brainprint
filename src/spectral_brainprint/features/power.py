"""Frozen 108-column power baseline; quality flags remain separate."""
import numpy as np
from spectral_brainprint.constants import CHANNELS
from spectral_brainprint.spectra.welch import band_power

def features_from_psd(psd, frequencies, config):
    """Return 54 log-power + 54 relative-power values, channel-major per block."""
    bands = config["bands"]
    absolute = np.stack([band_power(psd, frequencies, b["low_hz"], b["high_hz"])
                         for b in bands], axis=-1)
    denominator = band_power(psd, frequencies, *config["relative_denominator_hz"])
    floor = float(config["numerical_power_floor"])
    invalid = denominator <= floor
    log_power = np.log10(np.maximum(absolute, floor))
    relative = np.divide(absolute, denominator[..., None], out=np.zeros_like(absolute),
                         where=denominator[..., None] > floor)
    values = np.concatenate([log_power.reshape(*log_power.shape[:-2], -1),
                             relative.reshape(*relative.shape[:-2], -1)], axis=-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite power features")
    return values, invalid


def feature_dictionary(config):
    rows = []
    for block in ["log10_power", "relative_power"]:
        for channel in CHANNELS:
            for band in config["bands"]:
                rows.append({"column": len(rows), "name": f"{block}__{channel}__{band['name']}",
                             "block": block, "channel": channel, "band": band["name"],
                             "low_hz": band["low_hz"], "high_hz": band["high_hz"],
                             "unit": "log10(native_amplitude^2)" if block == "log10_power" else "dimensionless"})
    return rows
