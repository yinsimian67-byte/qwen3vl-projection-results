#!/usr/bin/env python3
"""Create a reproducible teaching subset; never mix official data splits."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

OFFICIAL_FILES = {
    "train": "highlight_train_release.jsonl",
    "dev": "highlight_val_release.jsonl",
    "final": "highlight_test_with_gt.jsonl",
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def source_id(vid):
    parts = vid.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid QVHighlights vid: {vid}")
    float(parts[1]); float(parts[2])
    return parts[0]


def assert_disjoint(splits):
    seen_qids, seen_sources = {}, {}
    for name, rows in splits.items():
        local = set()
        for row in rows:
            qid = row["qid"]
            if qid in local or qid in seen_qids:
                raise ValueError(f"Duplicated/overlapping qid {qid} in {name}")
            local.add(qid)
            seen_qids[qid] = name
            source = source_id(row["vid"])
            if source in seen_sources and seen_sources[source] != name:
                raise ValueError(f"Source video {source} crosses splits")
            seen_sources[source] = name


def select_subset(rows, count, seed):
    """Shuffle source groups, then queries; exact count, no cross-split reuse."""
    if not 1 <= count <= len(rows):
        raise ValueError(f"Requested {count} queries from {len(rows)} rows")
    groups = defaultdict(list)
    for row in rows:
        groups[source_id(row["vid"])].append(row)
    rng = random.Random(seed)
    sources = sorted(groups)
    rng.shuffle(sources)
    selected = []
    for source in sources:
        group = sorted(groups[source], key=lambda row: row["qid"])
        rng.shuffle(group)
        selected.extend(group[:count - len(selected)])
        if len(selected) == count:
            break
    return sorted(selected, key=lambda row: row["qid"])


def prepare(official_dir, output_dir, counts, seed=42):
    official_dir, output_dir = Path(official_dir), Path(output_dir)
    originals = {split: read_jsonl(official_dir / name) for split, name in OFFICIAL_FILES.items()}
    assert_disjoint(originals)
    selected = {split: select_subset(rows, counts[split], seed + offset)
                for offset, (split, rows) in enumerate(originals.items())}
    assert_disjoint(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "seed": seed, "benchmark": "course subset, not full QVHighlights",
                "selection": "shuffle original source groups and queries; take requested count",
                "protocol": "train=official train; dev=official val; final=official public test GT; no source overlap",
                "final_ground_truth_public": True, "splits": {}}
    for split, rows in selected.items():
        labeled = [dict(row, split=split) for row in rows]
        write_jsonl(output_dir / f"{split}.jsonl", labeled)
        unlabeled = [{key: row[key] for key in ("qid", "query", "duration", "vid")} | {"split": split}
                     for row in rows]
        write_jsonl(output_dir / f"{split}_queries.jsonl", unlabeled)
        original_path = official_dir / OFFICIAL_FILES[split]
        manifest["splits"][split] = {
            "queries": len(rows), "clips": len({row["vid"] for row in rows}),
            "source_videos": len({source_id(row["vid"]) for row in rows}),
            "gt_intervals": sum(len(row["relevant_windows"]) for row in rows),
            "official_file": original_path.name,
            "official_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
            "qid_sha256": hashlib.sha256(json.dumps([row["qid"] for row in rows]).encode()).hexdigest(),
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", default="data/official")
    parser.add_argument("--output-dir", default="data/course")
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--dev-size", type=int, default=200)
    parser.add_argument("--final-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.official_dir, args.output_dir,
          {"train": args.train_size, "dev": args.dev_size, "final": args.final_size}, args.seed), indent=2))


if __name__ == "__main__":
    main()
