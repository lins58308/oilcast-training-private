#!/usr/bin/env python3
"""Preserve the exact numeric experiment inputs without article text or keys.

The original artifact remains the source for signed-Java comparisons. This
small deterministic archive contains everything consumed by load_frozen for
the Python experiments, with derivation hashes for the original artifact.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def freeze(root,variant,output):
    root=Path(root);manifest_raw=(root/'manifest.json').read_bytes()
    original=json.loads(manifest_raw)
    if original['variant']!=variant:raise ValueError('Wrong frozen input variant')
    for name,expected in original['sha256'].items():
        if Path(name).name!=name:raise ValueError('Unsafe frozen input filename')
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed frozen input: '+name)
    dataset=json.loads((root/'dataset.json').read_bytes())
    required=('origin','targetDate','cutoff','features','basePrices','nextWeekDeltas')
    dataset['examples']=[{**{k:r[k] for k in required},
        'news':[{k:a[k] for k in ('source','publishedAt','availableAt')} for a in r['news']]}
        for r in dataset['examples']]
    files={'dataset.json':(json.dumps(dataset,ensure_ascii=False,separators=(',',':'))+'\n').encode()}
    for name in ('moea-daily.csv','dubai-monthly.csv','policy.json'):files[name]=(root/name).read_bytes()
    manifest={**original,'sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},
        'numericResearchSnapshot':True,'articleTextAndHeadlinesExcluded':True,
        'sourceManifestSha256':hashlib.sha256(manifest_raw).hexdigest(),'originalSourceManifest':original}
    files['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(target,'w:xz',preset=6) as archive:
        for name,raw in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0;info.mode=0o644
            archive.addfile(info,io.BytesIO(raw))
    print(json.dumps({'variant':variant,'examples':len(dataset['examples']),'archiveBytes':target.stat().st_size,
        'archiveSha256':hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',required=True)
    parser.add_argument('--variant',choices=('v6','v7'),required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();freeze(args.inputs,args.variant,args.output)
