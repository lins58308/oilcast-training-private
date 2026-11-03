# OilCast CPC absorption forecast experiments — 2026-11-03

## Official verifiable data

- Dated CPC historical fuel press releases: https://www.cpc.com.tw/news2.aspx?n=28
- Retail adjustment mechanism: https://price.ndc.gov.tw/p/zh_tw/energy
- CPC actual calculation table: https://vipmbr.cpc.com.tw/mbwebs/showpriceajust.aspx
- MOEA daily Dubai/Brent/WTI open CSV: https://www2.moeaea.gov.tw/oil111/csv/GlobalOilOpenData.csv

Archived a reproducible original CPC history snapshot at data/research/cpc-policy-20261103.json, sourced from each original public CPC News_Content.aspx page.

**Verified coverage:** 2019-11-03 through 2026-10-31. v6 has 364 notices, 215 parsed original theoretical adjustments, 249 parsed total-absorption figures; v7 has 365 notices, 217 theoretical, 250 absorption. Each version has 44 special absorption notices. Values are attributed to their individual official URLs.

Conservative source-time assumption: each date-only notice is available no earlier than **23:59:59 Asia/Taipei on its displayed publication date**. Original first-publication second and retrospective edits are NOT independently certified. Future Sunday target announcements are explicitly excluded. Missing subsidy data is masked, not assumed zero. The extracted 'absorptionGas' amount represents the total if a separate tax-relief component is included; never double-subtract tax and total.

## Decomposition and tests

The model distinguishes historical crude-pressure features (70% Dubai/30% Brent, FX, other global market values), prior CPC cost changes, prior policy theoretical changes, company/neighbor absorption, government excise tax reductions, special holiday no-rise and freeze notices.

- Existing Android public 26-feature model remains unchanged; all 12 new policy inputs per fuel live in a reproducible research pipeline only.
- CatBoost research includes policy+news, policy+real official daily mix and policy-informed freeze gating, with five separated pre-audit validation periods.
- Frozen multilingual ~0.3B E5 text encoder plus MLP (v6) or temporal Transformer (v7) was actually retrained on original CPC policy state. Tests separate core training, early stopping, selection and 13-week blind audit.
- A distinct Saturday pre-Sunday-announcement nowcast can consume later spot data; it is a two-day product, NOT equivalent to the Monday seven-day forecast.
- All headline, policy, and market features are time-censored. Hypothetical original historical market publishing times remain an estimation, not certified vintages.

## Out-of-time audit results (NTD per liter, mean of four fuels)

| Model | Historical five-fold MAE | Locked 13-week MAE |
|---|---:|---:|
| No-change baseline | 0.2169 | **0.0519** |
| Original price+market | 0.1927 | 0.0750 |
| v6 policy+MOEA | **0.1912** | 0.1654 |
| v7 policy+market | **0.1919** | 0.1000 |
| v6 Saturday nowcast+policy | 0.2131 | 0.1173 |
| v7 Saturday nowcast+policy | 0.2142 | 0.1269 |

These results do not demonstrate real out-of-time gains from the new policy-aware model. No research candidate is eligible to replace the signed Android production model.

The separate deep policy runs also selected the no-change expert for all four fuel types, despite successful real news embeddings and model training.

Primary run evidence:
- v6 historical policy: https://github.com/lins58308/oilcast-training-private/actions/runs/41559298876
- v7 historical policy: https://github.com/lins58308/oilcast-training-private/actions/runs/41559307646
- v6 deep training: https://github.com/lins58308/oilcast-training-private/actions/runs/41560335891
- v7 deep training: https://github.com/lins58308/oilcast-training-private/actions/runs/41560344849
- v6 late nowcast: https://github.com/lins58308/oilcast-training-private/actions/runs/41560793648
- v7 late nowcast: https://github.com/lins58308/oilcast-training-private/actions/runs/41560800565

## Next defensible move

Deploy no new predictive candidate until improved multiple-era out-of-time metrics are observed. Policy-regime shifts can make the next CPC announced retail price impossible to infer just from crude trends. Investigate **vintage-verified CPC specific weekly 7D3B FX index and contemporaneous tax/neighbor ceilings**, then distinguish live announced price (authoritative, not ML prediction) from days-ahead forecast.

The 2019–2026 CPC policy archive is permanent reproducible source data and can support future incremental retraining as genuinely new realized CPC weeks become available.
