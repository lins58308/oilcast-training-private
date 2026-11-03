#!/usr/bin/env python3
"""Compare signed-core candidate diagnostics without claiming promotion."""
import argparse
import hashlib
import json
from pathlib import Path

FUELS = ("92", "95", "98", "diesel")


def comparison(original, proposed, variant):
    before, after = original["periods"], proposed["periods"]
    if original["weeks"] != proposed["weeks"] or len(before) != len(after) or len(after) != 8:
        raise ValueError("Not the same historical experiment")
    heads = {}
    for family in (4, 6) if variant == "v7" else (4,):
        fuels = {}
        for f, fuel in enumerate(FUELS):
            periods = []
            for old, new in zip(before, after):
                if any(old[key] != new[key] for key in ("start", "end", "role")):
                    raise ValueError("Historical periods changed")
                b = next(h for h in old["heads"] if h["family"] == family)["fuels"][f]
                a = next(h for h in new["heads"] if h["family"] == family)["fuels"][f]
                if b["weeks"] != a["weeks"] or abs(b["baselineMae"]-a["baselineMae"]) > 1e-9:
                    raise ValueError("Official outcomes or lead time changed")
                changed, total = a["changedWeeks"], a["weeks"]
                recall, false = a["changedWeekRecall"], a["falseMoveRate"]
                direction = ((changed*(recall or 0)) + (total-changed)*(1-(false or 0))) / total
                periods.append({"role":new["role"],"start":new["start"],"end":new["end"],
                    "beforeMae":b["mae"],"afterMae":a["mae"],"baselineMae":a["baselineMae"],
                    "changedWeekRecall":recall,"falseMoveRate":false,"changedWeeks":changed,
                    "directionAccuracy":direction,"baselineDirectionAccuracy":(total-changed)/total})
            audit = periods[-1]
            recent = periods[-3:-1]
            validation = (all(p["afterMae"]+.015 < p["baselineMae"]-1e-9 for p in recent)
                          if variant == "v6" else
                          recent[-1]["afterMae"]+.01 < recent[-1]["baselineMae"]-1e-9
                          and recent[-1]["directionAccuracy"] >= recent[-1]["baselineDirectionAccuracy"]-.05
                          and recent[-1]["falseMoveRate"] is not None and recent[-1]["falseMoveRate"] <= .25)
            fuels[fuel] = {
                "earlierFoldBeforeMae":sum(p["beforeMae"] for p in periods[:5])/5,
                "earlierFoldAfterMae":sum(p["afterMae"] for p in periods[:5])/5,
                "audit":audit,"originalValidationGatesPassedAtFixedRegularizer":bool(validation),
                "auditBeatsFlatBy0_01":audit["afterMae"]+.01 < audit["baselineMae"]-1e-9,
                "periods":periods,
            }
        heads[str(family)] = {"fuels":fuels,
            "allFourAuditErrorsLowerThanOriginal":all(q["audit"]["afterMae"] < q["audit"]["beforeMae"]-1e-9 for q in fuels.values()),
            "allFourOriginalGatesPassedAtFixedRegularizer":all(q["originalValidationGatesPassedAtFixedRegularizer"] and q["auditBeatsFlatBy0_01"] for q in fuels.values())}
    return {"schema":1,"variant":variant,"heads":heads,
        "candidateDiagnosticsOnly":True,"fitFrozenBeforeEachBlock":True,
        "fixedRegularizers":{"4":3.,"6":.01} if variant=="v7" else {"4":3.},
        "auditPreviouslyInspected":True,"automaticPromotionApproved":False,
        "note":"Lower error than an old candidate does not establish superiority to no-change or approval of the selected active model."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True)
    p.add_argument("--proposed",required=True)
    p.add_argument("--variant",choices=("v6","v7"),required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    paths=[Path(args.original),Path(args.proposed)]
    result=comparison(*(json.loads(p.read_text()) for p in paths),args.variant)
    result["reportSha256"]={label:hashlib.sha256(path.read_bytes()).hexdigest()
                           for label,path in zip(("original","proposed"),paths)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({"variant":args.variant,"heads":{h:{
        "allFourOriginalGatesPassed":d["allFourOriginalGatesPassedAtFixedRegularizer"],
        "auditMae":{f:[r["audit"]["beforeMae"],r["audit"]["afterMae"],r["audit"]["baselineMae"]]
                    for f,r in d["fuels"].items()}} for h,d in result["heads"].items()}},indent=2))


if __name__=="__main__":
    main()
