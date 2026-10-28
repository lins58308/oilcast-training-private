# OilCast deep neural v7: validation and safe rollout

**Status: staged prototype, not deployed to the public Android feed.**

The existing production feed still publishes `multisource-adaptive-v6`. An unmodified
Android 3.3.12 rejects the new `multisource-neural-v7` contract. Do **not** merge
this branch and run scheduled cloud publication before the compatible Android
release has been rolled out. Deploy a separate v7 feed or arrange a versioned
dual feed for users who cannot upgrade; retain the signed v6 download endpoint.

## Model

- Candidate family 6 is a two-hidden-layer MLP: 26 inputs, 8 tanh units, 4 tanh
  units, 1 output estimating the next weekly pump-price change in TWD.
- Uses the same public historical fuel prices, lagged FRED prices/exchange rates,
  and available-at-time news context as the existing multi-source model.
- 156-week training window, recent-weighted Adam optimizer, regularization,
  chronological internal early stopping; deterministic weight initialization.
- Upweights weeks in which the official price actually changed so the network
  learns a movement signal rather than always returning an unchanged price.
- JSON stores validated network parameters; the Android Java core performs
  inference without mobile learning, model training, or external ML runtimes.

## Backtesting and safeguards

- **39 weeks rolling-origin validation**: for week T, only prices and source
  snapshots available through T-1 may enter the fit; a smaller window is used
  when the full historical dataset is short.
- **13 weeks locked holdout**: shown in the report, but not used to choose the
  model or tune its parameters. This avoids using the final test to pick a winner.
- Validation compares absolute error, direction hit rate, changed-price recall,
  and false move rate. A candidate may have up to TWD 0.045 worse validation
  MAE than the flat baseline if compensated by directional benefit; a restricted
  neural exploration mode permits close-MAE NN candidates with observable
  movement detection.
- Actual cloud accuracy percentages continue to be written in the signed
  `cloudAccuracy` history. They must never be replaced by historical backtests.
- The 13-week holdout is an honest audit. Aggressive NN selection may have
  worse holdout MAE and direction accuracy; this is **not** hidden or relabeled
  as high accuracy. If performance is inadequate, refine the method and rerun
  chronological tests before public deployment.

## Current offline validation

- Demo training with public historical fixture: 349 weeks, revision 1,
  `multisource-neural-v7`, completed without private signing keys.
- `NeuralLearnerTest`: deterministic training, parameter serialization,
  response to price-history signals, malformed weights rejection.
- `CloudContractTest`: 54 cloud contract assertions, including RSA verification
  and inference parity. Existing Python publisher tests: 14 passing.
- This demo is dated 2026-09-13 and is **not** a live model or a production
  accuracy measurement.

## Rollout checklist

1. Build and sign Android **3.3.13** (versionCode 23), test a v6 signed download
   and a v7 signed download; ensure both use the pinned RSA public key.
2. Make a separately versioned v7 feed available without changing the v6 URLs,
   then pilot v7 on an Android 3.3.13 test build.
3. Collect actual target-week outcomes and compare to the v6 baseline using
   cloud-signed MAE, direction hit rate and false move rate.
4. If metrics pass explicit release criteria, deploy Android update and switch
   only the compatible clients to the v7 feed. Keep old clients working.

**Do not put GitHub PATs, model signing private keys or Google Play upload
keystores in this source tree, in an Android bundle or in the public ZIP.**
