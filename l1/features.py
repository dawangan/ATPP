"""
Stage 4: event table -> covariates z_i for each duration.

Predictability rule: the covariate for x_i = t_i - t_{i-1} may only use events
with effective time tau <= t_{i-1}. Using t_i instead leaks the event into the
interval it is supposed to predict.

Feature (per event kind k):
    z_{i,k} = sum_{j in k, tau_j <= t_{i-1}} 2^{-(t_{i-1} - tau_j) / h}
h is a half-life in seconds. The decay is a modelling choice, not a fact;
profile h over a grid rather than fixing it (see scripts/run_layer1.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decayed_event_feature(t_prev_ns: np.ndarray, event_ns: np.ndarray, half_life_s: float,
                          cutoff_half_lives: float = 20.0) -> np.ndarray:
    half_life_ns = half_life_s*1e9
    span = int(half_life_ns*cutoff_half_lives)
    z = np.zeros(len(t_prev_ns))
    for tau in event_ns: 
        lo = np.searchsorted(t_prev_ns,tau,side="left")
        hi = np.searchsorted(t_prev_ns, tau+span, side ="right")
        elapsed = (t_prev_ns[lo:hi]- tau).astype(float)
        decay = np.exp2(-elapsed/half_life_ns) 
        z[lo:hi]+=decay
    return z
        
        
    


def build_design(dur: pd.DataFrame, events: pd.DataFrame, kinds=("scheduled", "surprise"),
                 half_life_s: float = 1800.0, time_col: str = "eff_ns") -> pd.DataFrame:
    """Columns z_<kind> aligned with dur rows (dur must be sorted by ts_prev)."""
    t = dur["ts_prev"].to_numpy(np.int64)
    if np.any(np.diff(t) < 0):
        raise ValueError("dur must be sorted by ts_prev")
    out = {}
    for k in kinds:
        ev = events.loc[events["kind"] == k, time_col].to_numpy(np.int64)
        out[f"z_{k}"] = decayed_event_feature(t, ev, half_life_s)
    return pd.DataFrame(out, index=dur.index)
