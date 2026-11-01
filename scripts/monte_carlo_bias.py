"""Monte Carlo: is the event estimator biased? Simulate the market from known truth,
run the full pipeline (prepare -> events -> diurnal -> Log-ACD), record estimates."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from l1 import simulate, trades, edgar, diurnal, features, acd
from multiprocessing import Pool

def one(seed, level_sd=0.08):
    df, pages, truth = simulate.simulate_market(n_days=60, n_scheduled=10, n_surprise=10, seed=seed, level_sd=level_sd)
    dur = trades.prepare(df)
    days = pd.DatetimeIndex(sorted(dur["date"].unique()))
    ev = edgar.build_event_table(edgar.filings_frame(pages), days, "utc")
    m = diurnal.fit_diurnal(dur, exclude_dates=ev["eff_date"].unique())
    adj = diurnal.adjust(dur, m)
    Z = features.build_design(adj, ev, half_life_s=1800)
    a = pd.concat([adj, Z], axis=1)
    r = acd.fit(acd.ACDData.from_frame(a, zcols=Z.columns), zcols=list(Z.columns))
    out = {"seed": seed, "level_sd": level_sd, "beta": r.theta[2]}
    for k in ("scheduled", "surprise"):
        i = r.names.index(f"gamma_z_{k}")
        lr = acd.long_run_effect(r, f"gamma_z_{k}")
        out[f"g_{k}"] = r.theta[i]; out[f"se_{k}"] = r.se_robust[i]; out[f"lr_{k}"] = lr["effect"]; out[f"lrse_{k}"] = lr["se"]; out[f"cse_{k}"] = r.se_cluster[i]
    return out

if __name__ == "__main__":
    jobs = [(s, 0.08) for s in range(300, 312)]
    with Pool(2) as p:
        res = p.starmap(one, jobs)
    d = pd.DataFrame(res)
    d.to_csv(sys.argv[2], index=False)
    for lsd, g in d.groupby("level_sd"):
        print(f"\nlevel_sd={lsd}  n={len(g)}   truth: gamma sched -0.04, surp -0.03; beta 0.92; long-run sched -0.50, surp -0.375")
        for c in ["beta", "g_scheduled", "g_surprise", "lr_scheduled", "lr_surprise"]:
            print(f"  {c:14s} mean={g[c].mean():+.4f}  sd across sims={g[c].std():.4f}")
        for k in ("scheduled", "surprise"):
            print(f"  {k}: mean robust SE of gamma={g['se_'+k].mean():.4f}  vs MC sd={g['g_'+k].std():.4f}  ratio={g['g_'+k].std()/g['se_'+k].mean():.2f}  | cluster SE={g['cse_'+k].mean():.4f} ratio={g['g_'+k].std()/g['cse_'+k].mean():.2f}")
