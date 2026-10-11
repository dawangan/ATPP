"""
Stage 5: Log-ACD(1,1) with exogenous event covariates, exponential or Weibull errors.

Model (Bauwens & Giot 2000, "Log-ACD1", plus covariates):
    x_i = psi_i * eps_i,     E[eps_i] = 1,  eps iid
    l_i = log psi_i = omega + alpha * log x_{i-1} + (beta - alpha) * l_{i-1} + gamma' z_i

Rewrite with log x_{i-1} = l_{i-1} + log eps_{i-1}:
    l_i = omega + beta * l_{i-1} + alpha * log eps_{i-1} + gamma' z_i
so beta is the persistence of log psi, alpha the news-impact, and |beta| < 1 is
covariance stationarity of l. No positivity constraints are needed anywhere,
which is why the log form is used instead of linear EACD: in linear EACD a
negative gamma can push psi below zero on event windows.

Why the recursion is fast: l_i - (beta-alpha) l_{i-1} = omega + alpha log x_{i-1} + gamma'z_i
is a first-order linear filter with a KNOWN input sequence (log x is data), so
scipy.signal.lfilter computes it in C. Days are separate series: l is reset to
l0 at each day's first duration, which is conditioned on, not scored.

Errors:
  'exp'     : eps ~ Exp(1)              ll_i = -l_i - x_i e^{-l_i}
  'weibull' : eps ~ Weibull(k) scaled to mean 1, scale s_i = lam_k psi_i, lam_k = 1/Gamma(1+1/k)
              ll_i = log k - log s_i + (k-1)(log x_i - log s_i) - (x_i/s_i)^k
The exponential QMLE is consistent for (omega, alpha, beta, gamma) even when the
true eps is not exponential (Engle & Russell 1998; Drost & Werker 2004), but then
the classical LR statistic is NOT chi-square. Use the robust (sandwich) Wald test
for gamma; the LR test is valid only under the correct error law.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, signal, special, stats


@dataclass
class ACDData:
    x: np.ndarray            # adjusted durations, > 0
    Z: np.ndarray            # (N, K) covariates, K may be 0
    day_start: np.ndarray    # bool (N,), True at each day's first duration
    l0: float = 0.0          # reset value of log psi at day starts

    @classmethod
    def from_frame(cls, d, zcols=(), day_col="date", x_col="x"):
        x = d[x_col].to_numpy(float)
        Z = d[list(zcols)].to_numpy(float) if len(zcols) else np.zeros((len(d), 0))
        day = d[day_col].to_numpy()
        start = np.r_[True, day[1:] != day[:-1]]
        return cls(x=x, Z=Z, day_start=start, l0=float(np.log(np.mean(x))))

    @property
    def K(self):
        return self.Z.shape[1]


def param_names(K: int, dist: str, zcols=None):
    g = [f"gamma_{c}" for c in zcols] if zcols else [f"gamma_{j}" for j in range(K)]
    return ["omega", "alpha", "beta"] + g + (["log_k"] if dist == "weibull" else [])


def log_psi(theta: np.ndarray, data: ACDData) -> np.ndarray:
    omega, alpha, beta, g = theta[0], theta[1], theta[2], theta[3:3+data.K]
    b = beta-alpha
    lx = np.log(data.x) 
    l = np.empty(len(data.x))
    for i in range(len(l)): 
        if data.day_start[i]:
            l[i] = data.l0
        else:
            l[i] = omega + alpha*lx[i-1] + b*l[i-1] + (data.Z[i] @ g if data.K != 0 else 0.0)
    return l 
    


def loglik_obs(theta: np.ndarray, data: ACDData, dist: str = "exp") -> np.ndarray:
    l  = log_psi(theta = theta, data= data)
    keep = ~data.day_start
    x,l = data.x[keep], l[keep]
    
    if dist == "exp":
        l_ret = -l-x*np.exp(-l)
        return l_ret
    elif dist == "weibull":
        k=np.exp(theta[-1])
        lam = 1/special.gamma(1+1/k)
        ls = l+np.log(lam)
        r = np.log(x)-ls
        k_ret = np.log(k)-ls+(k-1)*r-np.exp(k*r)
        return k_ret
    else: 
        raise ValueError("dist must be either exp or weibull.")
        return None

@dataclass
class FitResult:
    theta: np.ndarray
    names: list
    dist: str
    loglik: float
    n: int
    converged: bool
    cov_robust: np.ndarray | None = None
    cov_opg: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def se_robust(self):
        return np.sqrt(np.diag(self.cov_robust)) if self.cov_robust is not None else None

    def table(self):
        import pandas as pd
        se = self.se_robust
        t = self.theta / se if se is not None else np.full_like(self.theta, np.nan)
        return pd.DataFrame({"estimate": self.theta, "se_robust": se, "t": t,
                             "p": 2 * stats.norm.sf(np.abs(t))}, index=self.names)


def fit(data: ACDData, dist: str = "exp", theta0=None, zcols=None, robust_se: bool = True,
        maxiter: int = 500) -> FitResult:
    K = data.K
    p = 3 + K + (dist == "weibull")
    if theta0 is None:
        theta0 = np.r_[0.0, 0.05, 0.9, np.zeros(K), [0.0] * (dist == "weibull")]
    bounds = [(None, None), (-0.99, 0.99), (-0.999, 0.999)] + [(None, None)] * K + \
             ([(np.log(0.05), np.log(20.0))] if dist == "weibull" else [])

    def nll(th):
        v = -np.mean(loglik_obs(th, data, dist))
        return v if np.isfinite(v) else 1e10

    res = optimize.minimize(nll, np.asarray(theta0, float), method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": maxiter})
    th = res.x
    n = int((~data.day_start).sum())
    out = FitResult(theta=th, names=param_names(K, dist, zcols), dist=dist,
                    loglik=float(-res.fun * n), n=n, converged=bool(res.success))
    if robust_se:
        out.cov_robust, out.cov_opg = sandwich_cov(th, data, dist)
    return out


def _steps(theta):
    return 1e-4 * np.maximum(1.0, np.abs(theta))


def sandwich_cov(theta: np.ndarray, data: ACDData, dist: str):
    p,h  = len(theta), _steps(theta) 
    n = len(loglik_obs(theta,data,dist))
    cols = []
    
    for j in range(p): 
        up = theta.copy()
        up[j] +=h[j] 
        down = theta.copy()
        down[j] -=h[j] 
        col = (loglik_obs(up, data, dist) - loglik_obs(down, data, dist)) / (2 * h[j])
        cols.append(col) 
        
    S = np.column_stack(cols)
    B = S.T@S/n 
        
    def m(th): 
        return np.mean(loglik_obs(th, data, dist))

    def m_shift(*shifts):                 # shifts are (index, amount) pairs
        th = theta.copy()
        for idx, amt in shifts:
            th[idx] += amt
        return m(th)

    m0 = m(theta)
    H = np.zeros((p, p))
    for j in range(p):
        H[j, j] = (m_shift((j, h[j])) - 2 * m0 + m_shift((j, -h[j]))) / h[j] ** 2
        for k in range(j + 1, p):
            H[j, k] = (m_shift((j, h[j]), (k, h[k]))
                       - m_shift((j, h[j]), (k, -h[k]))
                       - m_shift((j, -h[j]), (k, h[k]))
                       + m_shift((j, -h[j]), (k, -h[k]))) / (4 * h[j] * h[k])
            H[k, j] = H[j, k]
    A = -H

    # --- sandwich
    Ai = np.linalg.pinv(A)
    cov_robust = Ai @ B @ Ai / n
    cov_opg = np.linalg.pinv(B) / n
    return cov_robust, cov_opg
    
        


def residuals(res: FitResult, data: ACDData) -> np.ndarray:
    """eps_hat = x / psi_hat on scored observations. Should have mean ~1 and
    little autocorrelation if the mean equation is adequate."""
    l = log_psi(res.theta, data)
    return (data.x / np.exp(l))[~data.day_start]


# ---------------------------------------------------------------- tests

def lr_test(full: FitResult, restricted: FitResult, q: int) -> dict:
    stat = 2 * (full.loglik - restricted.loglik)
    return {"stat": float(stat), "df": q, "p": float(stats.chi2.sf(stat, q))}


def wald_test(res: FitResult, names) -> dict:
    idx = [res.names.index(n) for n in names]
    b = res.theta[idx]
    V = res.cov_robust[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.solve(V, b))
    return {"stat": stat, "df": len(idx), "p": float(stats.chi2.sf(stat, len(idx)))}


def long_run_effect(res: FitResult, name: str) -> dict:
    ig = res.names.index(name)
    ib = res.names.index("beta")
    g, b = res.theta[ig], res.theta[ib]
    eff = g / (1 - b)
    grad = np.array([1 / (1 - b), g / (1 - b) ** 2])
    V = res.cov_robust[np.ix_([ig, ib], [ig, ib])]
    se = float(np.sqrt(grad @ V @ grad))
    return {"effect": float(eff), "se": se, "multiplier": float(np.exp(eff))}

def ljung_box(r: np.ndarray, lags: int = 20) -> dict:
    r = np.asarray(r, float) - np.mean(r)
    n = len(r)
    denom = r @ r
    ac = np.array([(r[k:] @ r[:-k]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(ac ** 2 / (n - np.arange(1, lags + 1)))
    return {"Q": float(q), "df": lags, "p": float(stats.chi2.sf(q, lags)), "acf": ac}


# ---------------------------------------------------------------- simulation

def simulate(n: int, theta, Z=None, dist: str = "exp", day_len: int | None = None, rng=None,
             l0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate adjusted durations from the model. Returns (x, day_start).
    Z must be (n, K) if gamma is non-empty. day_len splits the sample into days."""
    rng = np.random.default_rng(rng)
    theta = np.asarray(theta, float)
    omega, alpha, beta = theta[:3]
    Z = np.zeros((n, 0)) if Z is None else np.asarray(Z, float)
    K = Z.shape[1]
    g = theta[3:3 + K]
    if dist == "exp":
        eps = rng.exponential(1.0, n)
    else:
        k = np.exp(theta[-1])
        eps = rng.weibull(k, n) / special.gamma(1 + 1 / k)
    start = np.zeros(n, bool)
    start[::(day_len or n)] = True
    x = np.empty(n)
    l = l0
    for i in range(n):
        if start[i]:
            l = l0
        else:
            l = omega + alpha * np.log(x[i - 1]) + (beta - alpha) * l + (Z[i] @ g if K else 0.0)
        x[i] = np.exp(l) * eps[i]
    return x, start
