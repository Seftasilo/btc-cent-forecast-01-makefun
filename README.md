# btc-cent-forecast-01

Forecast Bitcoin's US-dollar price 4 hours ahead from the last 72 hours of
1-minute closes (make.fun competition `btc-cent-forecast-01`).

## Model

`main.py` (Python 3.13, stdlib only: `json`, `math`, `os`) reads one case
(`/input/<id>.in`, 4,320 closes) and writes one guess
(`/output/<id>.out`): `{"forecasts":[{"case":"<id>","price":<number>}]}`.

Guess = checkpoint close × exp(predicted 4h log-return), where the predicted
log-return is the average of two signals, clipped to ±0.5%:

1. **Ridge regression** (lambda=1000, no intercept), fit at run time inside
   the case itself. One sample every 5 minutes: 17 features from the 48
   hours ending at t (log-returns over 5–2880 min, log deviations from the
   60/240/1440-min SMAs, realised volatility over 240 min, 2 trend
   accelerations), target the log-return from t to t+240. Features are
   z-scored on the training samples, so the fit adapts to the case's own
   regime.
2. **Mean reversion** to the 24-hour mean:
   −0.10 × (log close − log mean of the last 1,440 closes).

Any failure (short history, singular fit, bad data) falls back to
persistence (the checkpoint close), so a case always stays valid. The
program is deterministic: two runs on the same case write identical bytes.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in ~0.04 s per case (limit 8 s).

## Practice results (p00–p05)

| case | checkpoint close | prediction |
|------|------------------|------------|
| p00 | 74764.91526 | 74691.83199 |
| p01 | 74447.80384 | 74501.21087 |
| p02 | 73680.83594 | 73887.21476 |
| p03 | 73210.50202 | 73223.08484 |
| p04 | 73777.35178 | 73707.66824 |
| p05 | 73545.88170 | 73535.95744 |

Every prediction differs from its checkpoint close (the model is active,
not persistence), two runs are byte-identical, and all six cases run in
~0.25 s total.
