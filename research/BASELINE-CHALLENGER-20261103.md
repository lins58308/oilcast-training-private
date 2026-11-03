# Can the unchanged-price baseline be improved for every fuel?

## Mechanism and implementation

Predicting a smaller nonzero movement every week does not solve a freeze:
in a 13-week sample with one 0.7 increase, always predicting zero has MAE
0.05385. Always predicting +0.1 has MAE 0.13846. An accepted correction must
save more error on changed weeks than it introduces on unchanged weeks.
For the existing 0.01 TWD/L improvement margin, it must additionally save
more than 0.01 times the number of evaluated weeks in total absolute error.

`baseline_challenger.py` estimates an empirical conditional outcome
distribution, then chooses its median, including zero as a possible action.
It compares a local kernel distribution and five fixed forest configurations.
One variant requires estimated absolute-loss savings to exceed one estimated
standard error; this is an abstention heuristic, not a confidence guarantee.
All fitting includes the full available history, with a 0.5 recency-weight
floor. Gasoline shares a correction but every official fuel label is scored.
The forecast horizon remains Monday to the next Monday, seven days.

The continuous replay also tests a fixed online guard: enable a correction
only if the prior 26 settled weeks beat no change by 0.005 TWD/L for each
fuel in its group. The current outcome never determines its own forecast.
This guard is a candidate policy, not a replacement promotion threshold.

## Shared policy-feature corrections

The old absorption-change calculation could walk past an equal previous
amount to an older different amount. It also returned zero when the current
notice was not frozen, and counted notices across missing or duplicate weeks.
The common `policy_features.py` now compares adjacent publication weeks,
independently of whether the current price changed. Missing weeks stop the
sequence and duplicate notices in one publication week count once.

Before this fix, 188 of 250 comparable gasoline-policy origins differed from
the immediately preceding week's amount. The audit is retained in the result
summary. `policy_state.py` additionally separates total absorption, tax relief
and the known net CPC portion, with explicit missingness masks. Original
official notices are unchanged. The standard-library policy helper introduces
no optional ML dependency into production validation.

## Continuous replay, rather than a favorable subset

Frozen version-specific inputs remain from run 41573170767. The replay uses
1,139 consecutive target weeks, 2004-10-11 through 2026-08-03. At each 13-week
block, fitting occurs before the block; the online guard updates from past
settled outcomes weekly. These are retrospective diagnostics with the
original publication-lag assumptions, not blind or live performance.

| Fuel | No-change MAE | v6 guarded MAE | v7 guarded MAE |
|---|---:|---:|---:|
| 92 | 0.291835 | 0.291308 | 0.290694 |
| 95 | 0.292537 | 0.292011 | 0.291396 |
| 98 | 0.292976 | 0.292450 | 0.291835 |
| Diesel | 0.312643 | 0.312116 | 0.310887 |

All four averages are slightly lower, but the reductions are only
0.000527 TWD/L for v6 and 0.001141 / 0.001756 for v7 gasoline / diesel.
They are far below the existing 0.01 improvement margin. A paired circular
52-week block bootstrap (2,000 draws) has intervals spanning zero for every
fuel; this small average benefit does not establish a reliable improvement.
That bootstrap is descriptive because this history was already examined.

By era, guarded gasoline worsens in 2004–2018, improves in 2019–2025, and
matches no change in 2026 before the audit. This rules out a claim of uniform
improvement across regimes. The earlier 65-week diagnostic suggested about
6% improvement for a fixed cautious forest, but only one of its five windows
had a meaningful win. It must not substitute for the continuous replay.

## Upgrade status

The six new candidates are checked on five older 13-week folds and the
original recent validation periods. Choices are written and hashed before
the previously examined final 13 weeks are scored. No candidate clears all
consistency, MAE, direction and false-movement checks. Both versions retain
no change in that audit: gasoline MAE 0.05385 and diesel 0.04615.

**All-four upgrade remains NOT PASSED.** Original margins and the signed
Android inference interface remain unchanged. The workflow preserves both
new and existing experiment evidence, then rejects an accuracy upgrade if
neither complete selected package clears the four separate audit margins.

The missing information is material: the official 2026-09-14 adjustment
notice describes the 2026-09-07–09-11 oil-price surge and was published on
2026-09-12, after the 2026-09-07 Monday origin. Those later quotes and that
policy decision cannot be inserted into the earlier forecast. Source:
https://www.moea.gov.tw/mns/populace/news/News.aspx?kind=1&menu_id=40&news_id=123955

Future progress needs better genuinely available pre-announcement signals,
original market vintages, and prospective frozen forecasts. Retuning the
known September jump cannot provide fresh verification.

## Reproduction

Install `research/requirements-mae.txt`, recover the version's original
frozen input artifact, then run:

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m unittest discover -s research/mae_tests
python3 research/baseline_challenger.py --inputs INPUT_DIRECTORY --variant v6 --output output/baseline-challenger --continuous-replay
```

Use the v7 artifact and `--variant v7` for that source. Reports include every
historical candidate, rejected gate, weekly replay prediction, loss budget
and paired uncertainty estimate. The shared policy fix is also applied when
the original eight-candidate laboratory is rerun; its prior result snapshots
remain historical evidence for their original source revisions.
