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
