"""Create self-excluding SHA-256 manifest for the public numerical archive.

Run once after changing any file, then verify with verify_release.py and verify_revision.py.
Excludes private/, docs/internal/, .git/, .venv/, __pycache__/, .pytest_cache/, and .DS_Store.
"""
from pathlib import Path
import argparse, hashlib, json

EXCLUDED_DIRS = {'.git', '__pycache__', '.venv', '.pytest_cache', '.ipynb_checkpoints', 'private'}


def hashfile(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(root):
    root = Path(root).resolve()
    dest = root / 'docs/RELEASE_SHA256.json'
    entries = {}
    for f in sorted(root.rglob('*')):
        if not f.is_file() or f == dest:
            continue
        rel = f.relative_to(root)
        if f.name == '.DS_Store' or f.name.startswith('Icon\r'):
            continue
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if rel.as_posix().startswith('docs/internal/'):
            continue
        entries[rel.as_posix()] = {'sha256': hashfile(f), 'bytes': f.stat().st_size}
    manifest = {
        'status': 'PUBLIC_NUMERICAL_PACKAGE_FILE_INVENTORY',
        'note': 'This is a hash manifest of file bytes, not evidence of human data validity or public publication.',
        'self_excluded': 'docs/RELEASE_SHA256.json',
        'file_count': len(entries),
        'files': entries,
    }
    dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print('Wrote', dest, len(entries), 'files')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.')
    args = p.parse_args()
    run(args.root)
