# OilCast accuracy benchmark: real Dubai/Brent (2026-11-03)

## Evidence and data provenance

The Taiwan government official dataset [data.gov.tw 6029](https://data.gov.tw/dataset/6029)
provides 6,894 observed-day rows, dating from 2000-01-03 to October 2026,
and includes WTI, Dubai and Brent daily spot prices. Its canonical CSV URL is
https://www2.moeaea.gov.tw/oil111/csv/GlobalOilOpenData.csv .

The experiment reads the real official prices and reconstructs a 70% Dubai +
30% Brent indicator, comparing the week ending before each CPC training origin
with the previous week. Conservative eligibility delay: source day must be at
least 3 calendar days before the *Taiwan-local CPC Monday*, not its preceding
UTC Sunday. A handful of genuine N/A missing quotes are excluded; historical
negative WTI values are accepted instead of falsified. The reconstructed
indicator is **not** necessarily identical to CPC's licensed Platts 7D3B
series and does not recreate the original price *publication vintages*.
A separate IMF/FRED monthly Dubai (POILDUBUSDM) is embargoed 75 days after
month end, evaluated independently.

The dataset contains 1,412 supervised CPC weekly origins and 1,383 weekly
origins where both MOEA Dubai and Brent have sufficient quotes.

## Verified chronological results

Five non-overlapping pre-audit folds (13 weeks each), same historical
CPC prices, 4 fuels, 1-decimal NTD/L predictions. Folds are used to select
candidate families. Final **13 weeks** are never used to select or tune.

| Model family | pre-audit average MAE | latest 13-week audit MAE |
|---|---:|---:|
| Unchanged-price baseline | 0.2169 | 0.0519 |
| CPC price-only CatBoost | 0.1946 | 0.0596 |
| CPC + existing Brent/WTI/FX | **0.1927** | 0.0750 |
| CPC + market + delayed monthly Dubai | 0.1946 | 0.0769 |
| CPC + market + government daily 70/30 | 0.2050 | 0.0962 |
| CPC + market + 70/30, 50% forecast shrink | 0.2127 | 0.0596 |
| CPC + market, 25% forecast shrink | 0.2165 | **0.0519** |

The only reason some shrunken forecasts match unchanged-price baseline
on the late holdout is that they round to the same 0.1 NTD/L prediction.
They are WORSE on historical validation and must not be selected using
the last 13 weeks. No proposed new family beats the baseline on the
independent late audit. **Production promotion is NOT approved.**

v6 actual full run:
https://github.com/lins58308/oilcast-training-private/actions/runs/41556715443

v7 actual full run:
https://github.com/lins58308/oilcast-training-private/actions/runs/41556734606

v6 and v7 source history, tests and signatures are kept separate; this
research does NOT alter the signed Android model input contract.

## Next material input missing

CPC's official price-adjustment table
https://vipmbr.cpc.com.tw/mbwebs/showpriceajust.aspx
includes theoretical retail price, neighbor-country limit and government
absorption amounts, e.g. October–November 2026. Their exact historically
*known-at-forecast-time* vintages must be assembled before forecasting
the recent policy-driven absorption regime. Never train on a Sunday price
announcement to predict that same next-Monday posted CPC price: that
would leak the known label, not demonstrate forecasting skill.
