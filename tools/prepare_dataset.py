"""Check a labelled image dataset and prepare sequence-safe YOLO splits.

This offline utility never trains a model or creates labels from predictions.
CSV columns: image,label,sequence,split. Paths are relative to the CSV file.
Leave every split empty for an approximate 70/20/10 sequence-group split,
or provide train/val/test for every row to preserve a reviewed partition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SPLITS = ('train', 'val', 'test')


def read_labels(path: Path, classes: list[str]) -> list[int]:
    if not path.is_file():
        raise ValueError(f'Missing label file: {path}. An empty file must explicitly mark a negative image.')
    ids = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'{path}:{line_no}: expected class, x_center, y_center, width, height')
        values = list(map(float, fields))
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f'{path}:{line_no}: non-finite label value')
        cls, cx, cy, w, h = values
        if cls != int(cls) or not 0 <= cls < len(classes):
            raise ValueError(f'{path}:{line_no}: class ID is outside the class list')
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'{path}:{line_no}: YOLO coordinates must be normalised; box size must be positive')
        ids.append(int(cls))
    return ids


def prepare(manifest: Path, output: Path, classes: list[str], source_url: str,
            license_name: str, seed: int = 42, materialize: bool = False) -> dict:
    manifest, output = manifest.resolve(), output.resolve()
    if output.exists():
        raise ValueError('Output already exists. Choose a new directory; source files will not be overwritten.')
    if not classes or len(set(classes)) != len(classes) or any(not c.strip() for c in classes):
        raise ValueError('Supply a non-empty, unique class list in the intended class-ID order.')
    if not source_url.strip() or not license_name.strip():
        raise ValueError('Record the source URL and license before preparing data.')
    rows = []
    seen_paths = set()
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'image', 'label', 'sequence'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV must contain image,label,sequence; split is optional.')
        for line_no, raw in enumerate(reader, 2):
            if any(not (raw.get(k) or '').strip() for k in ('image', 'label', 'sequence')):
                raise ValueError(f'CSV row {line_no}: image, label and sequence are required.')
            image = (manifest.parent / raw['image']).resolve()
            label = (manifest.parent / raw['label']).resolve()
            if image in seen_paths:
                raise ValueError(f'Duplicate image path in manifest: {image}')
            seen_paths.add(image)
            if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                raise ValueError(f'Unsupported image format: {image}')
            with Image.open(image) as im:
                im.verify()
            with Image.open(image) as im:
                size = list(im.size)
                # Compare decoded pixels, not just file bytes or filenames.
                digest = hashlib.sha256(str(im.size).encode() + im.convert('RGB').tobytes()).hexdigest()
            ids = read_labels(label, classes)
            split = (raw.get('split') or '').strip().lower()
            if split and split not in SPLITS:
                raise ValueError(f'CSV row {line_no}: split must be train, val or test.')
            rows.append(dict(image=str(image), label=str(label), sequence=raw['sequence'].strip(),
                             split=split, sha256_pixels=digest, size=size, class_ids=ids))
    if not rows:
        raise ValueError('Manifest contains no images.')

    # Sequences and exact duplicates form indivisible groups.
    parent = {r['sequence']: r['sequence'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(b)] = find(a)
    hashes = {}
    for r in rows:
        if r['sha256_pixels'] in hashes:
            union(r['sequence'], hashes[r['sha256_pixels']])
        else:
            hashes[r['sha256_pixels']] = r['sequence']
    groups = defaultdict(list)
    for r in rows:
        groups[find(r['sequence'])].append(r)
    labelled_splits = [bool(r['split']) for r in rows]
    if any(labelled_splits) and not all(labelled_splits):
        raise ValueError('Provide a split for every row, or leave every split empty.')
    if all(labelled_splits):
        for group in groups.values():
            if len({r['split'] for r in group}) != 1:
                raise ValueError('Data leakage: a sequence or duplicate image spans multiple splits.')
    else:
        keys = sorted(groups)
        if len(keys) < 3:
            raise ValueError('Need at least three independent sequence groups for train/val/test.')
        random.Random(seed).shuffle(keys)
        n_test, n_val = max(1, round(len(keys) * .1)), max(1, round(len(keys) * .2))
        n_train = len(keys) - n_test - n_val
        if n_train < 1:
            raise ValueError('Not enough independent groups to create all three splits.')
        for i, key in enumerate(keys):
            split = 'train' if i < n_train else 'val' if i < n_train + n_val else 'test'
            for r in groups[key]:
                r['split'] = split
    if set(r['split'] for r in rows) != set(SPLITS):
        raise ValueError('All three reviewed splits must contain images.')

    counts = Counter(c for r in rows for c in r['class_ids'])
    image_counts = Counter(c for r in rows for c in set(r['class_ids']))
    summary = dict(source_url=source_url, license=license_name, classes=classes,
                   images=len(rows), negative_images=sum(not r['class_ids'] for r in rows),
                   class_instances={name: counts[i] for i, name in enumerate(classes)},
                   class_images={name: image_counts[i] for i, name in enumerate(classes)},
                   split_images=dict(Counter(r['split'] for r in rows)),
                   independent_groups=len(groups), duplicate_images=len(rows) - len(hashes), seed=seed,
                   materialized=materialize,
                   note='Counts describe these input files, not model accuracy. No training was run.')
    output.mkdir(parents=True)
    with (output / 'checked_manifest.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image', 'label', 'sequence', 'split', 'sha256_pixels'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if materialize:
        for i, r in enumerate(rows):
            image_dir, label_dir = output / 'images' / r['split'], output / 'labels' / r['split']
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            name = f'{i:07d}'
            shutil.copy2(r['image'], image_dir / (name + Path(r['image']).suffix.lower()))
            shutil.copy2(r['label'], label_dir / (name + '.txt'))
        yaml = ('path: ' + json.dumps(output.as_posix()) + '\ntrain: images/train\nval: images/val\n'
                'test: images/test\nnames: ' + json.dumps(classes, ensure_ascii=False) + '\n')
        (output / 'data.yaml').write_text(yaml, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--classes', nargs='+', required=True)
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--license', dest='license_name', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--materialize', action='store_true', help='Copy checked images and labels into a new YOLO dataset.')
    args = parser.parse_args()
    try:
        result = prepare(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Data preparation stopped: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
