"""
Stage 3: SEC EDGAR 8-K filings -> timestamped, typed events on the trading clock.

Pipeline:
  fetch_submissions  -> raw JSON pages (recent + older "files" pages)
  filings_frame      -> one row per filing (form, items, acceptanceDateTime, ...)
  parse_acceptance   -> UTC timestamps under an explicit timezone assumption
  diagnose_acceptance_tz -> check that assumption against EDGAR's 06:00-22:00 ET window
  classify_items     -> scheduled / surprise / routine / other
  effective_market_time -> when the market could first trade on it (RTH clock)
  build_event_table  -> the table the feature builder consumes

Network calls need a real User-Agent: "Your Name your@email" (SEC fair-access
policy). Keep <= 10 requests/second.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .trades import RTH_CLOSE_S, RTH_OPEN_S, TZ

SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# ---------------------------------------------------------------- network (plumbing)

def _get_json(url: str, user_agent: str, sleep_s: float = 0.15):
    import requests
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=30)
    r.raise_for_status()
    time.sleep(sleep_s)
    return r.json()


def cik_for_ticker(ticker: str, user_agent: str) -> str:
    """10-digit zero-padded CIK for a ticker."""
    data = _get_json(TICKERS_URL, user_agent)
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return f"{int(row['cik_str']):010d}"
    raise KeyError(ticker)


def fetch_submissions(cik10: str, user_agent: str, cache_dir: str | Path | None = None) -> list[dict]:
    """All filing pages for a CIK: the `recent` block plus every older page listed
    under filings.files. Returns a list of column-oriented dicts (one per page).
    Caches raw JSON to cache_dir if given, so reruns are offline."""
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    def load(name):
        if cache and (cache / name).exists():
            return json.loads((cache / name).read_text())
        j = _get_json(SUBMISSIONS_URL.format(name=name), user_agent)
        if cache:
            (cache / name).write_text(json.dumps(j))
        return j

    top = load(f"CIK{cik10}.json")
    pages = [top["filings"]["recent"]]
    for f in top["filings"].get("files", []):
        pages.append(load(f["name"]))
    return pages


# ---------------------------------------------------------------- parsing

def filings_frame(pages: list[dict], forms=("8-K",)) -> pd.DataFrame:
    """Flatten column-oriented pages into rows, keep `forms` (exact match;
    8-K/A amendments are excluded by default because they re-report old news)."""
    cols = ["accessionNumber", "form", "filingDate", "reportDate", "acceptanceDateTime", "items", "primaryDocument"]
    frames = []
    for p in pages:
        n = len(p["accessionNumber"])
        frames.append(pd.DataFrame({c: p.get(c, [None] * n) for c in cols}))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("accessionNumber")
    df = df[df["form"].isin(list(forms))]
    df["items"] = df["items"].fillna("").astype(str)
    return df.reset_index(drop=True)


def parse_acceptance(raw: pd.Series, assume: str) -> pd.Series:
    if assume not in ("utc", "eastern"):
        raise ValueError(f"assume must be 'utc' or 'eastern', got {assume!r}")
    naive = pd.to_datetime(raw.str.rstrip("Z"), format="ISO8601")      # strip Z -> naive digits
    if assume == "utc":
        out = naive.dt.tz_localize("UTC")
    else:
        local = naive.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
        out = local.dt.tz_convert("UTC")
    return out.dt.as_unit("ns")


def diagnose_acceptance_tz(raw: pd.Series) -> pd.DataFrame:
    rows = []
    for assume in ("utc", "eastern"):
        local = parse_acceptance(raw, assume).dt.tz_convert(TZ)
        h = local.dt.hour + local.dt.minute / 60                        # fractional hour
        rows.append({
            "assume": assume,
            "in_edgar_window": ((h >= 6) & (h < 22)).mean(),
            "after_close_1600_1730": ((h >= 16) & (h < 17.5)).mean(),
            "pre_open_0600_0930": ((h >= 6) & (h < 9.5)).mean(),
            "n": len(raw),
        })
    return pd.DataFrame(rows)




# ---------------------------------------------------------------- taxonomy

DEFAULT_TAXONOMY = {
    # announced in advance, date known to the market
    "scheduled": {"2.02", "5.07"},
    # event date not known in advance
    "surprise": {"1.01", "1.02", "1.03", "1.05", "2.01", "2.03", "2.04", "2.05", "2.06",
                 "3.01", "4.01", "4.02", "5.01", "5.02", "8.01"},
    # carriers: exhibits, Reg FD furnishing, bylaw/charter amendments
    "routine": {"7.01", "9.01", "5.03", "5.05", "3.03"},
}
PRIORITY = ("scheduled", "surprise", "routine")


def classify_items(items: str, taxonomy: dict = DEFAULT_TAXONOMY) -> str:
    s = {p.strip() for p in items.split(",") if p.strip()}
    if s & taxonomy["scheduled"]:
        return "scheduled"
    if s & taxonomy["surprise"]:
        return "surprise"
    if s and s <= taxonomy["routine"]:
        return "routine"
    return "other"


def effective_market_time(accept_utc, trading_days, closes=None,
                          open_s=RTH_OPEN_S, close_s=RTH_CLOSE_S) -> pd.DataFrame:
    local = accept_utc.dt.tz_convert(TZ)
    midnight = local.dt.normalize()
    ldate = midnight.dt.tz_localize(None).dt.as_unit("ns")              # naive local date
    tod = (local - midnight).dt.total_seconds()                         # secs since local midnight

    days = (pd.DatetimeIndex(pd.to_datetime(pd.Index(trading_days)))
            .normalize().unique().sort_values().as_unit("ns"))

    is_td = ldate.isin(days)
    close_row = ldate.map(closes).fillna(close_s) if closes is not None else close_s

    intraday = is_td & (tod >= open_s) & (tod < close_row)
    pre_open = is_td & (tod < open_s)

    pos = days.searchsorted(pd.DatetimeIndex(ldate), side="right")      # first day strictly after
    has_next = pos < len(days)
    next_date = pd.Series(days[np.minimum(pos, len(days) - 1)].to_numpy(),
                          index=accept_utc.index).where(has_next)

    eff_date = ldate.where(intraday | pre_open, next_date)              # same day, else next day
    open_utc = ((eff_date + pd.to_timedelta(open_s, unit="s"))
                .dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
                .dt.tz_convert("UTC"))
    eff_utc = open_utc.where(~intraday, accept_utc)                     # intraday keeps its own time

    return pd.DataFrame({"eff_utc": eff_utc, "eff_date": eff_date, "intraday": intraday})



def build_event_table(filings: pd.DataFrame, trading_days, assume_tz: str,
                      closes: pd.Series | None = None, taxonomy: dict = DEFAULT_TAXONOMY) -> pd.DataFrame:
    """One row per 8-K inside the tick-data window:
    accession, items, kind, accept_utc, eff_utc, eff_ns (int64), eff_date, intraday."""
    f = filings.copy()
    f["accept_utc"] = parse_acceptance(f["acceptanceDateTime"], assume_tz)
    f["kind"] = f["items"].map(lambda s: classify_items(s, taxonomy))
    eff = effective_market_time(f["accept_utc"], trading_days, closes)
    f = pd.concat([f.reset_index(drop=True), eff.reset_index(drop=True)], axis=1)
    days = pd.DatetimeIndex(pd.to_datetime(pd.Index(trading_days))).normalize()
    f = f[f["eff_utc"].notna() & f["eff_date"].isin(days)]
    # a filing after the last in-sample day's close maps past the sample; also drop
    # filings accepted before the first in-sample day
    f = f[f["accept_utc"].dt.tz_convert(TZ).dt.tz_localize(None) >= days.min()]
    f["eff_ns"] = f["eff_utc"].dt.as_unit("ns").astype("int64")
    keep = ["accessionNumber", "items", "kind", "accept_utc", "eff_utc", "eff_ns", "eff_date", "intraday"]
    return f[keep].sort_values("eff_ns").reset_index(drop=True)
