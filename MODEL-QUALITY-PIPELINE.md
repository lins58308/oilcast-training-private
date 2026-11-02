# OilCast v6 / v7 chronological model process and promotion rules

Both versions follow the same **time-safe public-data and signed-model pipeline**.
The model is not promised to forecast every price change. Passing integration tests
means the pipeline is internally consistent, **not** that future forecasts are accurate.

## Common sequence

1. **Synchronize public observations:** CPC effective-date price-change events,
   retained CPC checkpoint, and public market/news context. A live run must fetch
   the CPC source successfully and refuses clearly stale events.
2. **Establish the information cutoff:** use `Asia/Taipei`; the last training
   label is the *current* Monday, never the upcoming Monday. A published
   upcoming price does not become a historical training label prematurely.
   Missing CPC change events mean "price unchanged", not "training failed".
3. **Construct only available-at-time features:** historical training snapshots
   are bounded by their historical origin; live market/news snapshot uses the
   current training time. Missing source availability must be explicit.
## Expanding training history: every observed week

- **No fixed 156-week training cutoff.** Every week that has sufficient
  historical lag features is included in the training dataset for every
  chronological fitting origin. The first 12 weeks are used for 12-week
  feature warm-up, not dropped for being old; the last observed week is a
  training *target* for the previous origin and a live feature origin.
- The most recent complete weekly observation can participate in **final
  retraining** after model selection. Validation and holdout *predictions*
  do not look into those later outcomes when evaluating their own origins.
- **Unchanged weeks are real labels** with a price movement of zero; they
  are retained even if CPC did not publish a new price-change event.
- For the recent-weighted robust regression and v7 neural preprocessing /
  Adam optimization, each older sample gets a recency weight of **at least
  0.5** of the newest unchanged week's base weight. Price-change emphasis
  is applied on top of that floor for neural training, never in place of
  inclusion.
- The JSON `trainingSamples` records **all history after feature warm-up**,
  rather than a cap of 156. `ModelTrainingCoverageTest` changes an early
  (pre-156-week) CPC row and proves that the final fitted parameters change.
  `model_quality_report.py` rejects unexpected shrinkage back to 156
  examples.
- More training data is not automatically better prediction: maintain
  rolling-origin and actual-outcome accuracy checks and reject regressions.

4. **Fit competing candidates on past observations:** v6 uses an unchanged-price baseline plus historical-price and
   multi-source ridge/regression alternatives; v7 additionally fits
   a compact deterministic neural net. All four fuels are evaluated separately.
5. **Time-ordered model comparison:** rolling-origin validation, then a later
   13-week audit. v6's existing 13-week activation gate is not an unbiased
   held-out estimate; v7 uses the earlier 39-week validation for selection,
   leaving the final 13 weeks untouched for audit.
6. **Inspect accuracy, not only job success:** compare MAE and RMSE with an
   unchanged-price baseline; separately report direction accuracy, recall when
   CPC actually changes prices, and the false-change rate. The neural candidate
   must beat the baseline on validation MAE without a large direction loss or
   false-change rate before being automatically selected in a v7 model package.
7. **Check signed App compatibility before publishing:** Python transport
   tests, CPC chronology/settlement tests, Java model contract/RSA/parity tests
   and (for v7) deterministic neural-weight tests. Production training performs
   this offline preflight before accessing the publisher.
8. **Publish and monitor:** a live signed release is uploaded immutably,
   hash-checked, and verified by the public Android/JVM downloader. The v6 stable
   channel and v7 test channel must remain separate. Existing Android clients
   must not receive an unsupported v7 schema.
9. **Settle only after target week starts:** on the day after the forecast's
   target Monday, compare the original forecast with the CPC effective price
   for that target date, even if there was no new CPC change row. Preserve the
   initial outcome and subsequent official corrections without double-counting.
   Signed live accuracy is tracked separately from historical validation.

## Full collected-news training, without look-ahead

- All unique, **safely dated** headlines and source metadata returned by the
  configured ten public news feeds and EIA archives are retained in the cloud
  training history. Oil-price keyword screening no longer discards other
  financial or geopolitical headlines. Unrelated stories count toward news
  activity, but never falsely assert an oil supply/demand shock.
- Historical feed metadata is not silently truncated after 90 days or after
  5,000 news items. A hard safety capacity of **20,000** retained reports
  currently prevents resource exhaustion; exceeding it fails the run rather
  than deleting the oldest reports. This is **not an unlimited internet news
  crawler** and does not download copyrighted article bodies.
- Every training origin computes its news features from the **preceding
  28 days of news already available at that point in time**, rather than
  looking at only 7 days or using upcoming headlines. A story discovered
  today but published years ago cannot be backdated into historical training
  as if the model saw it years ago.
- Recent-news feature weight is bounded below by **25%** over the 28-day
  context window. Equalized publisher statistics, six oil-economy event
  categories, and total news volume are still used by both v6 and v7.
- Each signed Android prediction payload contains only stories needed for
  its current inference window (at most 5,000). A separate immutable
  `training-state.json` release asset and mutable `training-checkpoint`
  retain the complete dated news archive. Recovery refuses to substitute a
  truncated Android payload when full training state cannot be restored.
- CI explicitly verifies news older than seven days, unrelated headlines,
  independent publishers, no future leakage, >5,000 historical headlines,
  signed-package prediction parity, and checkpoint restoration.
- Offline test success does not establish real-time source availability or
  superior predictions. Maintain signed Android compatibility and future
  realized-price accuracy gates before deployment.

## Installed Android compatibility blocker

The Java contract tests compile the **proposed shared core**, not an already
installed Android APK. Older APKs may still generate weekly model input with
`weekly(rows, nextMonday(today))`. Those APKs can reject the newly corrected
`weeklyThrough(rows, monday(today))` signed package, especially when CPC has
published a next-Monday effective price or there was no price change this week.
Before merging to the live feed, the Android application must adopt the new
as-of semantics and pass real-device signed download, prediction parity, and
rollback tests. A passing JVM CI check alone is **not** Android compatibility
evidence. Keep both PRs in draft and keep the existing stable and pinned test
releases unchanged until this compatibility work is complete.

## Automatic promotion is disabled

The quality report is a historical diagnostic. It deliberately sets
`automaticPromotionApproved=false`. The publishing workflow may only be
enabled under the existing repository/channel policies; test success alone does
not authorize switching a live App from stable v6 to v7. Before any production
v7 promotion, require Android device compatibility, an audited true out-of-time
set, and realized official CPC forecast-vs-outcome performance (MAE, direction,
changed-week recall, and false alarms).

## Tests

- `python3 -m unittest discover -s tests -p 'test_*.py'`
- `python3 run_training.py --as-of 2026-09-13T13:00:00Z --output output`
- Compile and execute `tests/ModelTimelineTest.java` and
  `tests/CloudContractTest.java`; run `tests/NeuralLearnerTest.java` on v7.
- `python3 tests/model_quality_report.py output/model.json`

The fixture runs are historical and intentionally unsigned demo models.
Do not insert model private keys, API tokens, or personal App usage data.
