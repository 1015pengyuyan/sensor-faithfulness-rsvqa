"""Public-schema export from a caller-supplied private source manifest.

The source manifest is local only: [{"source": "...", "target": "...jsonl",
"kind": "prediction|fixture|metadata", "expected_sha256": "..."}].
Do not commit this private source manifest. Raw scientific responses are never edited.
"""
import argparse
import json
from pathlib import Path
from public_io import sha, fixture, prediction, metadata, assert_public

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--sources',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
provenance=[]
for item in json.loads(a.sources.read_text(encoding='utf-8')):
    src=Path(item['source']);dst=(a.out/item['target']).resolve()
    if not dst.is_relative_to(a.out.resolve()):
        raise ValueError('Export target escapes output root')
    if dst.exists():raise FileExistsError('Public export never silently overwrites: '+item['target'])
    if sha(src)!=item['expected_sha256']:raise ValueError('Original source hash mismatch')
    dst.parent.mkdir(parents=True,exist_ok=True);n=0
    with src.open(encoding='utf-8-sig') as i,dst.open('w',encoding='utf-8',newline='\n') as o:
        for line in i:
            row=json.loads(line)
            transform={'prediction':prediction,'fixture':fixture,'metadata':metadata}[item['kind']]
            out=transform(row);assert_public(out)
            o.write(json.dumps(out,ensure_ascii=False,separators=(',',':'))+'\n');n+=1
    provenance.append(dict(original_name=src.name,original_sha256=sha(src),public_path=item['target'],
                           public_sha256=sha(dst),bytes=dst.stat().st_size,records=n,kind=item['kind']))
(a.out/'EXPORT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
