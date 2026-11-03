# Four-fuel MAE optimization, 2026-11-03

## Inputs and chronology

Original v6 source: `993ae6e4b593d929f6522e2706bb7e675b6004ca`.
Original v7 source: `2b7fac70aced13e65df157a4882ad3de9a87fb02`.
Frozen inputs: https://github.com/lins58308/oilcast-training-private/actions/runs/41573170767

Each version has 1,412 supervised weeks, from 1999-10-11 through 2026-10-26
origins, with target prices through 2026-11-02. The common export cutoff is
2026-11-03T02:00:00Z. Every input is SHA256-checked. The 368 original CPC
policy notices remain. The versions have different retained news archives,
so their independently frozen artifacts must stay separate. Original market
publication-lag assumptions remain research limitations.

Five older 13-week folds start 2021-05-17, 2022-05-16, 2023-05-15,
2024-05-13, and 2025-05-12. Recent validation covers 2025-02-10–2025-11-03
and 2025-11-10–2026-08-03. The **previously inspected** audit covers
2026-08-10–2026-11-02. These dates refer to target Mondays.

## Training changes

Family 4 previously clipped inputs at +/-4 but still minimized squared
error. `AbsoluteLoss` refines the existing linear model with 12 bounded IRLS
steps and width 0.1 TWD/L, giving large residuals bounded influence. Every
labeled week remains in training; the **recency base weight** stays at least
0.5. Robust residual weighting is separate from sample age.

V7 additionally replaces squared-error neural gradients with smooth absolute
error, removes 2.5x oversampling of changed weeks, and uses uniform weekly
MAE for chronological early stopping. Regularizers and eligible families
are ranked by MAE, with direction as a tie break. Its original direction and
false-movement safeguards remain. Linear 26-feature and neural 26-8-4-1
inference/serialization formats remain unchanged.

## Paired signed-core candidate results

MAE is TWD/L. These are **candidate diagnostics**, not the selected active
model's accuracy or a new blind trial. Both trainers use the same checkpoint
and feature origins. Each fit is frozen before its entire scored block;
this differs from the publisher's weekly-refit activation evaluation.
Fixed regularizers are 3.0 for family 4 and 0.01 for neural family 6.

| Fuel | v6 family 4 before | v6 after | v7 neural before | v7 after | No-change baseline |
|---|---:|---:|---:|---:|---:|
| 92 | 0.11538 | 0.11538 | 0.09231 | 0.07692 | 0.05385 |
| 95 | 0.11538 | 0.11538 | 0.08462 | 0.05385 | 0.05385 |
| 98 | 0.11538 | 0.11538 | 0.09231 | 0.06154 | 0.05385 |
| Diesel | 0.13077 | 0.06923 | 0.16154 | 0.10000 | 0.04615 |

V7's linear robust head also improves all four audit errors: 92 from
0.14615 to 0.10769; 95/98 from 0.15385 to 0.11538; diesel from 0.17692 to
0.11538. V6's older-fold family-4 errors improve for all four fuels, but
its combined recent-validation diesel MAE changes from 0.31923 to 0.32051.
The paired reports preserve that regression.

## Other learners and gate outcome

`mae_walk_forward.py` compares eight fixed candidates: ridge, linear
conditional medians, absolute-error boosting and tree-leaf medians, with
past freeze duration and movement-frequency features. V7 adds lagged
contexts. Every estimator fits all available training history. Gasoline
shares one learned adjustment, but each of 92/95/98 must pass separately;
original exceptional fuel labels are never replaced when scoring. Selection
is written and hashed before calculating the final audit.

Before safeguards, the best v6 older-fold MAE is 0.19538 for gasoline and
0.19538 for diesel versus baselines 0.21846 / 0.21231. V7's best values are
0.19692 / 0.20000. These refer to different historical candidates, not one
approved package. None meets all consistency/direction/false-alarm checks.

**All-four upgrade: NOT PASSED.** The research selector retains no change,
with audit MAE 0.05385 for each gasoline grade and 0.04615 for diesel.
The audit contains one changed week and twelve unchanged weeks. A 92.31%
direction hit rate can therefore hide zero change-event recall.

V6's two 39-week 0.015 TWD/L margins and v7's 0.01 margin with direction /
false-movement safeguards are preserved. The explicit four-fuel workflow
uploads evidence, then fails with `ACCURACY_GATE_NOT_MET` if any selected
fuel misses the separate 0.01 TWD/L audit improvement requirement. Passing
contract tests alone cannot turn the accuracy gate green.

Tests cover outlier influence, old-history participation, neural median
behavior, future-label/scaler leakage, temporal chronology, per-grade
rejection, Java rounding, missing-event recall and input hashes. Existing
publishing and Java contract/history/leakage tests remain required.

Production still requires Android device validation and prospective frozen
forecasts issued before official target announcements. Repeated tuning on
the known 2026 audit cannot provide that evidence. No release is promoted.

## Reproduce

Use Python 3.12, `research/requirements-mae.txt`, JDK17 and the original
checksummed org.json dependency. Download the version's artifact from run
41573170767, then:

```sh
python3 -m pip install -r research/requirements-mae.txt
python3 -m unittest discover -s research/mae_tests -v
python3 research/mae_walk_forward.py --inputs INPUT_DIRECTORY --variant v6 --output output/mae-trial
```

V7 uses its artifact and `--variant v7`. The workflow also checks out the
pinned original trainer, runs `SignedCoreMaeAudit.java` on both sources,
and generates their paired per-fuel JSON comparison.


The [baseline-challenger follow-up](BASELINE-CHALLENGER-20261103.md) corrects
shared policy features and expands validation to 1,139 continuous weeks. Its
small aggregate improvements are not sufficient for the original upgrade gate.
