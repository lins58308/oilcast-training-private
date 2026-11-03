# OilCast four-fuel MAE gate: causally dated shared-gasoline experiment (2026-11-03)

## What changed

- One three-class (down, unchanged, up) **shared gasoline** change head, yielding
  the same adjustment for 92 / 95 / 98. Separate **diesel** head.
- Only if a candidate's predicted price-change probability exceeds a
  historical confidence threshold may it output a nonzero adjustment.
  Its price-change size is calculated from a past-only historical median.
- Compare 36 models (4 feature families × 3 training lengths × 3 confidence
  thresholds). Five past non-overlapping 13-week windows select the
  hyperparameters, not the latest previously inspected 13 weeks.
- The three gasoline types get ONE model-family choice selected using
  historical joint gasoline MAE. Diesel is selected independently.
- Official CPC historical news and subsidy records merged from v6/v7
  independent original-source crawls: **368 distinct dated notices**, with
  219 parseable theoretical adjustment entries and 253 absorption entries.
  Duplicate identities or conflicting original-page fields are rejected.
  The same 368-event dataset is used for both v6 and v7, avoiding the
  earlier reproducibility problem (364 vs 365 records).
- Found 5 historical gasoline weeks whose 92/95/98 adjustment amounts
  differed. They are reported as explicit exceptions, not erased.

## Results

Baseline is a legitimate 0 NTD/L change for each week. Reported MAE is
NTD/L, fuel-specific. It is NEVER sufficient to have three good
gasoline fuel heads and a worse diesel head.

| Fuel | historical baseline MAE | gasoline/diesel classifier MAE | improvement | 2026 13-week baseline MAE | model MAE |
|---|---:|---:|---:|---:|---:|
| 92 | 0.21846 | 0.20000 | 8.45% | 0.05385 | 0.05385 |
| 95 | 0.21846 | 0.20000 | 8.45% | 0.05385 | 0.05385 |
| 98 | 0.21846 | 0.20000 | 8.45% | 0.05385 | 0.05385 |
| Diesel | 0.21231 | 0.21231 (no candidate qualified) | 0% | 0.04615 | 0.04615 |

The most recent thirteen weeks contain **one price change** per fuel,
and the changes were not known from their earlier feature origin.
The models forecast no change for that week and match the no-change
baseline in the holdout. Their directional accuracy is 12/13
(92.31%), but change-event recall is **0%**. Neither v6 nor v7
achieved the target of lower MAE for **all four** fuels in the final
historical audit; doing otherwise requires genuinely predictive
pre-event signals, not threshold gaming.

The older 2026 holdout was examined multiple times in previous
research projects. It must be treated as diagnostic, **not an untouched
blind trial**. Prospective forecasts issued before future CPC
announcements are needed before declaring a real production improvement.

Official context: CPC 95/98 follow 92 gasoline change amounts,
diesel differs; official adjustment formula uses 70% Dubai +30% Brent,
FX, neighbor-price ceiling and stabilization absorption.
https://price.ndc.gov.tw/p/zh_tw/energy

The 2026-09-14 sudden adjustment was announced on 2026-09-12,
after the preceding Monday forecast origin.
https://www.moea.gov.tw/mns/populace/news/News.aspx?kind=1&menu_id=40&news_id=123955

## Recorded CI evidence

- v6 independently trained model, full 36 candidate evaluation:
  https://github.com/lins58308/oilcast-training-private/actions/runs/41563568975
- v7 independently trained model, full 36 candidate evaluation:
  https://github.com/lins58308/oilcast-training-private/actions/runs/41563579613
- All optional CatBoost dependencies and tests are restricted to the
  research CI, not needed for Android cloud public-model publishing.
- Signed Android 26-feature contract and public release channels
  remain unchanged.

## Promotion policy

Don't promote a model because it performed better on some historical
windows. Require (a) pre-registered per-fuel MAE improvement for all
four, (b) market-shock change recall and false-movement-rate reporting,
(c) future as-of frozen forecasts not subject to repeated tuning,
(d) unchanged signed Android compatibility and provenance.
