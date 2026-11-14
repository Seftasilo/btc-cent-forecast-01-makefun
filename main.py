#!/usr/bin/env python3
"""BTC 4-hour-ahead price forecast for make.fun btc-cent-forecast-01.

Reads one case at a time: 4,320 one-minute BTC/USDC closes for the 72 hours
up to a checkpoint, and writes one guess for the close 240 minutes later.

Model (pure Python 3.13 stdlib: json, math, os):
  guess = checkpoint_close * exp(predicted 4h log-return)
  predicted log-return = clip(0.5 * ridge + 0.5 * mean_reversion, +/-0.5%)

  (1) Ridge regression (lambda=1000, no intercept), fit at run time inside
      the case itself. One sample per 5 minutes: features from the 48 hours
      ending at t, target the log-return from t to t+240. 17 features:
        - log-returns over 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440
          and 2880 minutes (11)
        - log deviations of the close from the 60/240/1440-minute simple
          moving averages (3)
        - realised volatility: std of 1-minute log-returns over the last
          240 minutes (1)
        - 2 trend accelerations: the log-return over the window just before
          the latest move, at two horizons -- (30m minus 5m) and
          (120m minus 30m) -- i.e. the trend excluding the freshest minutes
      Features are z-scored on the training samples before the fit, so the
      fit adapts to the case's own regime; nothing is carried over from any
      practice window.
  (2) Mean reversion to the 24-hour mean:
        -0.10 * (log close - log mean of the last 1,440 closes)

Any failure (too little history, singular fit, bad data) falls back to
persistence (the checkpoint close), so a case always stays valid.
Deterministic: the same input always produces the same bytes.
"""

import json
import math
import os

LAGS = [5, 10, 15, 30, 60, 120, 240, 480, 720, 1440, 2880]
SMA_WINDOWS = (60, 240, 1440)
HISTORY_MIN = 2880          # minutes of history each sample needs
SAMPLE_STEP = 5             # minutes between training samples
LAMBDA = 1000.0             # ridge penalty, intercept 0
MR_COEF = 0.10              # mean-reversion strength
CLIP = 0.005                # +/-0.5% clip on the predicted log-return
VOL_WINDOW = 240            # minutes for realised volatility


def _solve(a, b):
    """Solve a x = b by Gauss-Jordan with partial pivoting (small dense)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            raise ValueError("singular matrix")
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                fac = m[r][col] / pv
                for k in range(col, n + 1):
                    m[r][k] -= fac * m[col][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def _features(prices, t):
    """The 17 features at minute index t (prices[t] is the current close)."""
    p = prices
    f = [math.log(p[t] / p[t - k]) for k in LAGS]
    for k in SMA_WINDOWS:
        sma = sum(p[t - k + 1:t + 1]) / k
        f.append(math.log(p[t] / sma))
    rets = [math.log(p[i] / p[i - 1]) for i in range(t - VOL_WINDOW + 1, t + 1)]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    f.append(math.sqrt(var))
    # Trend accelerations: momentum just before the latest move.
    f.append(math.log(p[t] / p[t - 30]) - math.log(p[t] / p[t - 5]))
    f.append(math.log(p[t] / p[t - 120]) - math.log(p[t] / p[t - 30]))
    return f


def _ridge_predict(prices, horizon):
    """Fit the ridge on this case's own samples; predict at the checkpoint."""
    n = len(prices)
    ts = list(range(HISTORY_MIN, n - horizon, SAMPLE_STEP))
    if len(ts) < 10:
        raise ValueError("too few training samples")
    x = [_features(prices, t) for t in ts]
    y = [math.log(prices[t + horizon] / prices[t]) for t in ts]
    d = len(x[0])
    mu = [sum(row[j] for row in x) / len(x) for j in range(d)]
    sd = []
    for j in range(d):
        var = sum((row[j] - mu[j]) ** 2 for row in x) / len(x)
        s = math.sqrt(var)
        sd.append(s if s > 1e-12 else 1.0)
    xs = [[(row[j] - mu[j]) / sd[j] for j in range(d)] for row in x]
    xtx = [[0.0] * d for _ in range(d)]
    xty = [0.0] * d
    for row, yi in zip(xs, y):
        for a in range(d):
            xty[a] += row[a] * yi
            ra = row[a]
            for b in range(d):
                xtx[a][b] += ra * row[b]
    for a in range(d):
        xtx[a][a] += LAMBDA
    beta = _solve(xtx, xty)
    fc = _features(prices, n - 1)
    fcs = [(fc[j] - mu[j]) / sd[j] for j in range(d)]
    return sum(b * v for b, v in zip(beta, fcs))


def _mean_reversion(prices):
    close = prices[-1]
    mean24 = sum(prices[-1440:]) / 1440
    return -MR_COEF * (math.log(close) - math.log(mean24))


def forecast(prices, horizon):
    """Return the predicted price, or the checkpoint close on any failure."""
    close = prices[-1]
    try:
        if len(prices) < HISTORY_MIN + horizon + 1:
            raise ValueError("history too short")
        if not all(math.isfinite(v) and v > 0 for v in prices):
            raise ValueError("bad price data")
        ridge = _ridge_predict(prices, horizon)
        mr = _mean_reversion(prices)
        pred = 0.5 * ridge + 0.5 * mr
        pred = max(-CLIP, min(CLIP, pred))
        price = close * math.exp(pred)
        if not (math.isfinite(price) and 0 < price < 10_000_000):
            raise ValueError("invalid price")
        return round(price, 5)
    except Exception:
        return round(close, 5)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            case_id = case["case"]
            prices = case["markets"][0]["prices"]
            horizon = int(case.get("horizon_minutes", 240))
        except Exception:
            # A malformed case file must not kill the other cases' outputs.
            continue
        price = forecast(prices, horizon)
        out = {"forecasts": [{"case": case_id, "price": price}]}
        with open(os.path.join(out_dir, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
