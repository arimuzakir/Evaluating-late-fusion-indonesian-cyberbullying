"""Create self-excluding SHA-256 manifest for the public numerical archive.

Run once after changing any file, then verify and regenerate the ZIP.
"""
from pathlib import Path
import argparse,hashlib,json

def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def run(root):
    root=Path(root).resolve()
    dest=root/'docs/RELEASE_SHA256.json'
    entries={}
    for f in sorted(root.rglob('*')):
        if (not f.is_file() or f==dest or any(s in f.parts for s in ['.git','__pycache__','.venv'])):continue
        entries[f.relative_to(root).as_posix()]={'sha256':hashfile(f),'bytes':f.stat().st_size}
    manifest={'status':'PUBLIC_NUMERICAL_PACKAGE_FILE_INVENTORY',
              'note':'This is a hash manifest of file bytes, not evidence of human data validity or public publication.',
              'self_excluded':'docs/RELEASE_SHA256.json',
              'file_count':len(entries),'files':entries}
    dest.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('Wrote',dest,len(entries),'files')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');args=p.parse_args();run(args.root)
