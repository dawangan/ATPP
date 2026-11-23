"""Machine-readable pre-registration: frozen spec, explore/holdout split, drift and holdout guards."""
from __future__ import annotations

import pandas as pd

from .core import PKG, load_json, sha256_json

PREREG_PATH = PKG / "prereg.json"


class HoldoutViolation(RuntimeError):
    pass


def load(path=None) -> dict:
    return load_json(path or PREREG_PATH)


def prereg_hash(pr: dict) -> str:
    return sha256_json(pr)


def spec_params(params: dict, pr: dict) -> dict:
    """The subset of an experiment's params that defines the *specification* (what counts as a
    distinct look at the data). Seeds, output paths and compute knobs are excluded."""
    keys = pr["spec_keys"]
    return {k: params[k] for k in keys if k in params}


def drift(params: dict, pr: dict) -> list[str]:
    """Frozen-spec fields that this run overrides. A drifted run is allowed but is labelled
    exploratory: it cannot supply the pre-registered pass/fail verdict."""
    out = []
    for k, frozen in pr["frozen"].items():
        if k in params and params[k] != frozen:
            out.append(f"{k}={params[k]!r} (frozen: {frozen!r})")
    return out


def assert_explore(last_day, pr: dict) -> None:
    """Refuse anything that touches the holdout period."""
    if last_day is None:
        return
    if pd.Timestamp(last_day) > pd.Timestamp(pr["explore_end"]):
        raise HoldoutViolation(
            f"data reaches {pd.Timestamp(last_day).date()}, after explore_end={pr['explore_end']}. "
            "The holdout is human-only; the harness has no unlock.")
