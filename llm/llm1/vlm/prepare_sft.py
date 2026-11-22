"""Make training-only pointwise or pairwise examples from locator proposals.

Only GT windows from the supplied training annotation file are used. Evaluation
must retain the locator's original proposals; these augmented positives never
enter the evaluation candidate pool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from .common import file_sha256, join_inputs, load_jsonl, max_tiou, write_jsonl


def make_examples(annotation, candidate_row, windows, task, rng):
    if annotation.get("split") != "train" or candidate_row.get("split") != "train":
        raise ValueError("SFT preparation requires annotations and candidates explicitly marked split=train")
    ground_truth = annotation.get("relevant_windows", [])
    if not ground_truth:
        raise ValueError(f"No training GT windows for qid {annotation['qid']}")
    for start, end in ground_truth:
        if not 0 <= start < end <= float(annotation["duration"]):
            raise ValueError("Invalid training GT window")
    proposals = [window[:2] for window in windows]
    # Include one GT positive per query when the initial locator misses the event.
    # The provenance field makes this curriculum explicit and auditable.
    selected_gt = list(rng.choice(ground_truth))
    proposals.append(selected_gt)
    unique = []
    for clip in proposals:
        if clip not in unique:
            unique.append(clip)
    positives = [clip for clip in unique if max_tiou(clip, ground_truth) >= 0.5]
    negatives = [clip for clip in unique if max_tiou(clip, ground_truth) <= 0.1]
    common = {"qid": annotation["qid"], "vid": annotation["vid"],
              "query": annotation["query"], "task": task, "split": "train"}
    if task == "pointwise":
        # Per-query balancing prevents a weak locator from flooding SFT with No.
        rng.shuffle(positives)
        rng.shuffle(negatives)
        examples = []
        for clip, answer in [(clip, "Yes") for clip in positives[:1]] + [(clip, "No") for clip in negatives[:1]]:
            examples.append({**common, "clips": [clip], "answer": answer,
                             "source": "gt_augmented" if clip == selected_gt else "locator",
                             "target_tiou": max_tiou(clip, ground_truth)})
        return examples
    if task != "pairwise":
        raise ValueError(f"Unknown task {task}")
    if not positives or not negatives:
        return []
    positive, negative = rng.choice(positives), rng.choice(negatives)
    # Randomize positions; annotation IDs and candidate rank are never in prompt.
    clips, answer = ([positive, negative], "A") if rng.random() < 0.5 else ([negative, positive], "B")
    return [{**common, "clips": clips, "answer": answer, "source": "locator_and_gt",
             "target_tiou": [max_tiou(clip, ground_truth) for clip in clips]}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="TRAIN annotations only")
    parser.add_argument("--candidates", required=True, help="Predictions for the same TRAIN qids")
    parser.add_argument("--task", choices=["pointwise", "pairwise"], default="pointwise")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    joined = join_inputs(load_jsonl(args.annotations), load_jsonl(args.candidates))
    rng = random.Random(args.seed)
    rng.shuffle(joined)
    examples = []
    queries_without_negative = 0
    for annotation, candidate, windows in joined:
        rows = make_examples(annotation, candidate, windows, args.task, rng)
        if not any(max_tiou(window, annotation["relevant_windows"]) <= 0.1 for window in windows):
            queries_without_negative += 1
        examples.extend(rows)
    rng.shuffle(examples)
    examples = examples[:args.max_examples]
    if not examples:
        raise ValueError("No usable SFT examples; inspect training candidates and their tIoU distribution")
    write_jsonl(args.output, examples)
    manifest = {
        "task": args.task, "seed": args.seed, "examples": len(examples),
        "unique_qids": len({str(row["qid"]) for row in examples}),
        "source_query_count": len(joined), "queries_without_negative": queries_without_negative,
        "answer_counts": {label: sum(row["answer"] == label for row in examples) for label in ["Yes", "No", "A", "B"]},
        "annotation_file": str(Path(args.annotations).resolve()),
        "annotation_sha256": file_sha256(args.annotations),
        "candidate_file": str(Path(args.candidates).resolve()),
        "candidate_sha256": file_sha256(args.candidates),
        "warning": "Use official training annotations only; never include dev/final/test GT. GT-augmented positives are training-only.",
    }
    Path(args.output + ".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
