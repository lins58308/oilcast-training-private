# OilCast 2026-11-03 — Accuracy research protocol

## Root causes
1. CPC weekly pump prices do not equal world Brent prices. Official weekly
   indicator uses 70% Dubai + 30% Brent, USD/TWD FX, 80% pass-through,
   retail taxes, neighbor-price ceiling, smoothing and policy absorption.
   Sources: https://price.ndc.gov.tw/p/zh_tw/energy and
   https://vipmbr.cpc.com.tw/mbwebs/showpriceajust.aspx .
   OilCast's 26 inputs currently contain Brent, WTI, FX but NOT daily Dubai
   or official policy absorption. A daily 7D3B history/real-time archive is
   still needed before production-grade accurate mechanism reconstruction.
   IMF/FRED POILDUBUSDM supplies monthly Dubai history from 1992, not the
   daily Platts 7D3B and not historical real-time vintage releases.
2. Global headlines recorded retrospectively are not available as if they
   had been observed in 1999; embeddings do not automatically create
   predictive causality. Missing stories are not evidence of no conflict.
3. Short flat-price regimes in the latest holdout make a flat baseline hard
   to beat even if global crude moves sharply. Forecast policy constraints
   separately from crude; never hard-code a 2026 freeze into older periods.
4. Previously v6 used the most recent 13-week sample as a promotion gate.
   This leaked audit outcomes into selection. V6 now uses TWO non-overlapping
   39-week pre-audit windows. The last 13 weeks remain untouched until audit.

## Research benchmark
- Run research/accuracy_benchmark.py on 1,400+ dated weekly examples
  exported by SemanticResearchDataset.
- Equal time-point inputs: history, market quotes, policy-regime proxies,
  and 6 news-event topic channels.
- Compare pooled CatBoost price-only, price+market, price+market+news,
  and a zero-inflated move/no-move hurdle with the no-change baseline.
- Select solely on three pre-audit periods, then audit 13 untouched weeks.
- Report MAE, RMSE, move-direction accuracy, true-movement recall, and
  false movement rate for all four fuels.
- Never publish solely because ML training completes. Only deploy models
  that improve multiple pre-audit periods and holdout WITHOUT regressing
  app-model schema or verified signature. These CatBoost models cannot be
  deployed inside current Android apps without compatible inference, so
  research remains isolated.

Actual results are available as downloadable oilcast-accuracy-lab-* artifacts
in Actions, not predictions or new releases by default.
