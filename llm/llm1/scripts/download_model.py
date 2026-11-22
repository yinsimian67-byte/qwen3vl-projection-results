#!/usr/bin/env python3
"""Fetch pinned Qwen assets, verifying HF Git blob IDs or LFS SHA-256.

ModelScope is an optional weight mirror only. Metadata is always fetched from
the exact Hugging Face revision; mirror weight bytes must match that revision.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def digest(path, entry):
    is_lfs = 'lfs' in entry
    h = hashlib.sha256() if is_lfs else hashlib.sha1()
    if not is_lfs:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest() == (entry['lfs']['sha256'] if is_lfs else entry['blobId'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('downloads/qwen_model_manifest.json'))
    p.add_argument('--output', type=Path, default=Path('models/Qwen2.5-VL-3B-Instruct'))
    p.add_argument('--weights-source', choices=['huggingface', 'modelscope'], default='huggingface')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--metadata-only', action='store_true')
    mode.add_argument('--weights-only', action='store_true')
    args = p.parse_args()
    meta = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    entries = [f for f in meta['files'] if f['rfilename'] != '.gitattributes']
    if args.metadata_only:
        entries = [f for f in entries if not f['rfilename'].endswith('.safetensors')]
    if args.weights_only:
        entries = [f for f in entries if f['rfilename'].endswith('.safetensors')]
    def fetch(entry):
        name = entry['rfilename']
        dest = args.output / name
        if dest.exists() and dest.stat().st_size == entry['size'] and digest(dest, entry):
            print('Verified existing', name, flush=True)
            return
        is_weight = name.endswith('.safetensors')
        if is_weight and args.weights_source == 'modelscope':
            url = f'https://modelscope.cn/models/{meta["repo"]}/resolve/master/{name}'
        else:
            url = f'https://huggingface.co/{meta["repo"]}/resolve/{meta["revision"]}/{name}'
        partial = Path(str(dest) + '.part')
        for attempt in range(8):
            try:
                prefix = partial.stat().st_size if partial.exists() else 0
                req = urllib.request.Request(url, headers={'Range': f'bytes={prefix}-'} if prefix else {})
                if prefix < entry['size']:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        if prefix and response.status != 206:
                            raise IOError('Resume range ignored')
                        with partial.open('ab' if prefix else 'wb') as output:
                            while block := response.read(4 * 1024 * 1024):
                                output.write(block)
                if partial.stat().st_size != entry['size'] or not digest(partial, entry):
                    partial.rename(Path(str(partial) + f'.bad-{int(time.time())}'))
                    raise IOError('Byte count or source hash mismatch')
                partial.replace(dest)
                print('Downloaded and verified', name, entry['size'], flush=True)
                return
            except Exception as error:
                if attempt == 7:
                    raise
                print('Retry', name, str(error), flush=True)
                time.sleep(min(2 ** attempt, 20))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, entries))
    report = {'repo': meta['repo'], 'revision': meta['revision'],
              'verified_files': [f['rfilename'] for f in entries],
              'weights_source': args.weights_source, 'identity': 'HF Git blob / LFS SHA-256'}
    name = 'weights_verification.json' if args.weights_only else 'metadata_verification.json'
    (args.output / name).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
