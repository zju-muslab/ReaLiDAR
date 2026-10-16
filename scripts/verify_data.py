"""Verify the versioned scene data package without importing GPU libraries."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    failures = []
    for entry in manifest['files']:
        path = args.data_dir / entry['path']
        if not path.is_file():
            failures.append(f'Missing: {entry["path"]}')
        elif path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            failures.append(f'Checksum mismatch: {entry["path"]}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Verified {len(manifest["files"])} scene/GT files')


if __name__ == '__main__':
    main()
