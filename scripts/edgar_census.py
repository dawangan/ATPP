"""
8-K census: for a list of tickers, list every 8-K in a date window with its acceptance
time, item codes and trading-session bucket, then summarise how many TIMED events
(pre-open / intraday / after-hours, not just "next open") each ticker gives.

Free; needs only network access to sec.gov (SEC requires a contact in the User-Agent).

    python scripts/edgar_census.py --tickers MSFT AAPL NVDA --start 2023-09-19 --end 2025-09-18 \
        --user-agent "David Wang vonkluge112@gmail.com" --out census

    python scripts/edgar_census.py --ticker-file tickers.txt ...

Outputs (in --out):
    events.csv   one row per 8-K (accession, ticker, accepted_utc, accepted_et, bucket,
                 items, is_earnings, is_5_07, is_timed, event_day, primary_doc, filing_date, form)
    summary.csv  one row per ticker (counts by bucket and class)
    pull_plan.csv  ticker x event_day (the trading day whose tape the filing affects)

Buckets (America/New_York wall clock of the EDGAR acceptance time, which is genuine UTC):
    pre_open    04:00-09:30   extended-hours, filing time is a usable event time
    intraday    09:30-16:00   regular session
    after_hours 16:00-20:00   extended-hours, usable event time
    overnight   20:00-04:00 or non-trading day -> effect appears at the next 04:00/09:30 (weak timing)
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `l1` is importable
from l1 import edgar  # noqa: E402

ET = "America/New_York"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def get_json(url: str, ua: str, session: requests.Session, pause: float = 0.15):
    time.sleep(pause)                       # SEC fair-access limit is 10 req/s; stay well under
    r = session.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=30)
    r.raise_for_status()
    return r.json()


def ticker_to_cik(ua: str, session: requests.Session) -> dict[str, int]:
    data = get_json(SEC_TICKERS, ua, session)
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def filings_frame(block: dict) -> pd.DataFrame:
    cols = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "primaryDocument"]
    return pd.DataFrame({c: block.get(c, []) for c in cols})


def eight_ks_for(cik: int, ua: str, session: requests.Session, start: pd.Timestamp, forms: tuple[str, ...]  = ("8-K", "8-K/A")) -> pd.DataFrame:
    sub = get_json(SUBMISSIONS.format(cik=cik), ua, session)
    frames = [filings_frame(sub["filings"]["recent"])]
    # 'recent' holds the newest ~1000 filings; older ones sit in extra files. Fetch only if needed.
    if pd.Timestamp(frames[0]["filingDate"].min()) > start:
        for f in sub["filings"].get("files", []):
            if pd.Timestamp(f["filingTo"]) >= start:
                frames.append(filings_frame(get_json(f"https://data.sec.gov/submissions/{f['name']}", ua, session)))
    df = pd.concat(frames, ignore_index=True)
    return df[df["form"].isin(list(forms))].copy()



def bucket(acc_et: pd.Series) -> pd.Series:
    t = acc_et.dt.hour * 3600 + acc_et.dt.minute * 60 + acc_et.dt.second
    weekday = acc_et.dt.dayofweek < 5
    b = pd.Series("overnight", index=acc_et.index)
    b[weekday & (t >= 4 * 3600) & (t < 9.5 * 3600)] = "pre_open"
    b[weekday & (t >= 9.5 * 3600) & (t < 16 * 3600)] = "intraday"
    b[weekday & (t >= 16 * 3600) & (t < 20 * 3600)] = "after_hours"
    return b


def next_weekday(d: pd.Series) -> pd.Series:
    """Roll Saturday/Sunday forward to Monday. Exchange holidays are NOT modelled: the
    puller simply gets an empty day, which it skips."""
    d = d.dt.normalize()
    out = d.copy()
    wk = d.dt.dayofweek
    out[wk == 5] = d[wk == 5] + pd.Timedelta(days=2)
    out[wk == 6] = d[wk == 6] + pd.Timedelta(days=1)
    return out


def build_events(tickers: list[str], ua: str, start: str, end: str, tz_mode: str = "auto",
                 forms: tuple[str, ...] = ("8-K", "8-K/A")) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = requests.Session()
    cik_map = ticker_to_cik(ua, s)
    s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for tk in tickers:
        cik = cik_map.get(tk.upper())
        if cik is None:
            print(f"[skip] {tk}: not in SEC ticker file")
            continue
        try:
            df = eight_ks_for(cik, ua, s, s0, forms)
        except Exception as e:                      # keep going; report at the end
            print(f"[fail] {tk}: {e}")
            continue
        df["ticker"] = tk.upper()
        rows.append(df)
        print(f"[ok] {tk}: {len(df)} 8-K rows (all dates)")
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    ev = pd.concat(rows, ignore_index=True)

    ev["accepted_utc"] = pd.to_datetime(ev["acceptanceDateTime"], utc=True)
    # per-filer timezone fix (A1-4); the report is written to <out>/tz_report.csv in main()
    ev["accepted_utc"], tz_report = edgar.fix_acceptance_tz(
        ev["accepted_utc"], ev["filingDate"], group=ev["ticker"], mode=tz_mode)
    print(tz_report.to_string())

    ev["filing_date"] = ev["filingDate"]
    ev["accepted_et"] = ev["accepted_utc"].dt.tz_convert(ET)
    d = ev["accepted_et"].dt.tz_localize(None)
    ev = ev[(d >= s0) & (d < e0 + pd.Timedelta(days=1))].copy()
    ev["bucket"] = bucket(ev["accepted_et"])
    items = ev["items"].fillna("").str.replace(" ", "")
    ev["items"] = items
    ev["is_earnings"] = items.str.contains(r"(?:^|,)2\.02(?:,|$)")
    ev["is_5_07"] = items.str.contains(r"(?:^|,)5\.07(?:,|$)")
    ev["is_timed"] = ev["bucket"].isin(["pre_open", "intraday", "after_hours"])
    # trading day affected: the same day if filed before 20:00 on a weekday; otherwise the next weekday
    local_date = ev["accepted_et"].dt.tz_localize(None).dt.normalize()
    after_8pm = ev["accepted_et"].dt.hour >= 20
    ev["event_day"] = next_weekday(local_date.where(~after_8pm, local_date + pd.Timedelta(days=1)))
    ev = ev.rename(columns={"accessionNumber": "accession", "primaryDocument": "primary_doc"})
    keep = ["ticker", "accession", "accepted_utc", "accepted_et", "bucket", "items", "is_earnings",
            "is_5_07", "is_timed", "event_day", "primary_doc", "filing_date", "form"] 
    return ev[keep].sort_values(["ticker", "accepted_utc"]).reset_index(drop=True), tz_report


def summarise(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.assign(cls=lambda x: x["is_earnings"].map({True: "earnings", False: "other"}))
    piv = ev.pivot_table(index="ticker", columns=["cls", "bucket"], values="accession",
                         aggfunc="count", fill_value=0)
    piv.columns = [f"{c}_{b}" for c, b in piv.columns]
    piv["n_total"] = ev.groupby("ticker").size()
    piv["n_timed_all"] = ev[ev["is_timed"]].groupby("ticker").size()
    piv["n_timed_non_earnings"] = ev[ev["is_timed"] & ~ev["is_earnings"]].groupby("ticker").size()
    piv[["n_timed_all", "n_timed_non_earnings"]] = piv[["n_timed_all", "n_timed_non_earnings"]].fillna(0).astype(int)
    return piv.sort_values("n_timed_non_earnings", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--ticker-file")
    ap.add_argument("--start", default="2023-09-19")
    ap.add_argument("--end", default="2025-09-18")
    ap.add_argument("--user-agent", required=True, help='e.g. "Your Name you@example.com" (SEC requires it)')
    ap.add_argument("--out", default="census")
    ap.add_argument("--tz-mode", default="auto", choices=["auto", "utc", "shifted"])
    ap.add_argument("--forms", default="8-K,8-K/A",
                    help="comma list of form types to keep, e.g. 8-K,8-K/A,6-K")
    a = ap.parse_args()

    tickers = list(a.tickers)
    if a.ticker_file:
        tickers += [t.strip() for t in Path(a.ticker_file).read_text().split() if t.strip()]
    tickers = sorted(set(t.upper() for t in tickers))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    forms = tuple(f.strip() for f in a.forms.split(",") if f.strip())
    ev, tz_report = build_events(tickers, a.user_agent, a.start, a.end, a.tz_mode, forms=forms)
    if ev.empty:
        raise SystemExit("no events")
    print(ev.groupby(["form", "is_timed"]).size())
    ev.to_csv(out / "events.csv", index=False)
    summ = summarise(ev)
    summ.to_csv(out / "summary.csv")
    tz_report.to_csv(out/"tz_report.csv")
    plan = ev[["ticker", "event_day"]].drop_duplicates().sort_values(["ticker", "event_day"])
    plan.to_csv(out / "pull_plan.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== per-ticker summary ===")
    print(summ[["n_total", "n_timed_all", "n_timed_non_earnings"]].to_string())
    print(f"\nTOTAL timed events: {int(summ['n_timed_all'].sum())}  "
          f"timed non-earnings: {int(summ['n_timed_non_earnings'].sum())}  "
          f"(shape test wants >= ~100 of the latter)")


if __name__ == "__main__":
    main()
