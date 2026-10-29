# OilCast hybrid model experiment — 2026-10-29

**Shadow only. No signed production or testing release has been modified.**

Lightweight TFT-inspired feature selection and temporal attention, 1D CNN for
local signal extraction, parallel LSTM and GRU memory, sigmoid-gated residual
fusion, and a four-fuel regression head. This is **not full TFT**.

## Honest walk-forward testing
- Inputs: 12 weeks, CPC prices, lagged FRED market series, dated news summaries.
- Date and availability cutoffs prevent looking up future market/news observations.
- 26-week earlier validation (7 sampled origins) tunes two seeds and shrinkage.
- Final **13 weeks × 4 fuels = 52 cases** are out-of-sample, retrained by origin.
- Compare MAE, three-way direction, actual price-change recall and false moves.
- No leakage of the holdout into model or threshold selection.

### Local test report (historical seed)
- 327 complete weekly price rows.
- Hybrid: MAE **0.2923 TWD**, direction **63.46%**, actual-change recall **5%**.
- Price-unchanged baseline: MAE **0.2942 TWD**, direction **61.54%**.
- **Promotion gate failed**: the MAE difference is below the minimum 0.002 TWD.
- Historical data carries estimated availability. These metrics are not verified
  real-world forward performance and do not justify changing App predictions.

### Safeguards and next releases
- Read-only GitHub workflow uploads a shadow report as an artifact only.
- v6 production remains GitHub Releases latest; signed neural-v7 prerelease
  is an independent test feed with status-v7.json.
- Mobile open-test candidate 3.3.14 resolves v7 pointer, validates pinned RSA
  and payload hash, otherwise attempts stable v6.
- The hybrid cannot be pushed to phones until model parameters have a signed
  portable schema, matching JVM/Android inference, freshness checks, a staged
  rollout and performance confirmation.
- Google Play upload needs the existing private upload keystore, AAB signature
  and account access. Never commit these credentials.
