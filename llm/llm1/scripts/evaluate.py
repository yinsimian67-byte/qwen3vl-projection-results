#!/usr/bin/env python3
"""Strict fixed-three-candidate course evaluation using official MR metric functions."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "moment_detr"))
try:
    from .prepare_course_data import read_jsonl
except ImportError:
    from prepare_course_data import read_jsonl

TOP_K = 3


def tiou(first, second):
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = (first[1] - first[0]) + (second[1] - second[0]) - intersection
    return intersection / union if union > 0 else 0.0


def _index(rows, label):
    result = {}
    if not rows:
        raise ValueError(f"{label} is empty")
    for row in rows:
        qid = row.get("qid")
        if not isinstance(qid, int) or isinstance(qid, bool):
            raise ValueError(f"{label}: qid must be an integer")
        if qid in result:
            raise ValueError(f"{label}: duplicate qid {qid}")
        result[qid] = row
    return result


def validate_predictions(predictions, ground_truth, reference=None, top_k=TOP_K):
    pred, gt = _index(predictions, "predictions"), _index(ground_truth, "ground truth")
    if pred.keys() != gt.keys():
        raise ValueError(f"qid mismatch: missing={sorted(gt.keys()-pred.keys())[:10]}, extra={sorted(pred.keys()-gt.keys())[:10]}")
    if reference is not None:
        validate_predictions(reference, ground_truth, top_k=top_k)
        ref = _index(reference, "reference")
    for qid, row in pred.items():
        if row.get("vid") != gt[qid]["vid"]:
            raise ValueError(f"qid {qid}: vid mismatch")
        windows = row.get("pred_relevant_windows")
        if not isinstance(windows, list) or len(windows) != top_k:
            raise ValueError(f"qid {qid}: exactly {top_k} candidates required")
        for window in windows:
            if not isinstance(window, (list, tuple)) or len(window) != 3:
                raise ValueError(f"qid {qid}: each window must be [start,end,score]")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in window):
                raise ValueError(f"qid {qid}: nonfinite/non-numeric window")
            start, end, _ = window
            if not 0 <= start < end <= gt[qid]["duration"]:
                raise ValueError(f"qid {qid}: out-of-range or empty window {window}")
        if any(windows[i][2] < windows[i+1][2] for i in range(len(windows)-1)):
            raise ValueError(f"qid {qid}: scores must be in non-increasing rank order")
        if reference is not None:
            if Counter(tuple(window[:2]) for window in windows) != Counter(tuple(window[:2]) for window in ref[qid]["pred_relevant_windows"]):
                raise ValueError(f"qid {qid}: reranking changed the reference candidate pool/boundaries")
    return [pred[row["qid"]] for row in ground_truth]


def candidate_diagnostics(predictions, ground_truth, thresholds=(0.5, 0.7)):
    indexed = _index(predictions, "predictions")
    result = {}
    for threshold in thresholds:
        query_hits, coverage = [], []
        for row in ground_truth:
            matches = [any(tiou(candidate, truth) >= threshold for candidate in indexed[row["qid"]]["pred_relevant_windows"])
                       for truth in row["relevant_windows"]]
            query_hits.append(any(matches))
            coverage.append(sum(matches) / len(matches))
        result[f"QueryHit@3/tIoU{threshold}"] = round(100 * sum(query_hits)/len(query_hits), 4)
        result[f"GT-Coverage@3/tIoU{threshold}"] = round(100 * sum(coverage)/len(coverage), 4)
    return result


def evaluate(predictions, ground_truth, reference=None):
    from standalone_eval.eval import compute_mr_ap, compute_mr_r1, get_data_by_range
    predictions = validate_predictions(predictions, ground_truth, reference)
    for row in ground_truth:
        if not row.get("relevant_windows"):
            raise ValueError(f"qid {row['qid']}: missing ground truth windows")
    detailed = {}
    for name, bounds in (("full", [0, 150]), ("short", [0, 10]), ("middle", [10, 30]), ("long", [30, 150])):
        selected_pred, selected_gt = get_data_by_range(predictions, ground_truth, bounds)
        detailed[name] = ({"MR-mAP": compute_mr_ap(selected_pred, selected_gt, max_pred_windows=TOP_K, num_workers=0),
                           "MR-R1": compute_mr_r1(selected_pred, selected_gt), "queries": len(selected_gt)}
                          if selected_gt else {"MR-mAP": None, "MR-R1": None, "queries": 0})
    full = detailed["full"]
    return {
        "protocol": {"candidate_budget": TOP_K, "queries": len(ground_truth), "unit": "percentage (0-100)",
                     "official_metric_functions": "Moment-DETR standalone_eval.eval",
                     "iou_thresholds_mAP": "0.50:0.05:0.95", "same_candidate_pool_checked": reference is not None,
                     "scope": "course subset; not directly comparable with full benchmark top-10 results"},
        "brief": {"MR-full-mAP": full["MR-mAP"]["average"], "MR-full-mAP@0.5": full["MR-mAP"]["0.5"],
                  "MR-full-mAP@0.75": full["MR-mAP"]["0.75"], "MR-full-R1@0.5": full["MR-R1"]["0.5"],
                  "MR-full-R1@0.7": full["MR-R1"]["0.7"]},
        "candidate_diagnostics": candidate_diagnostics(predictions, ground_truth), "detailed": detailed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--reference-candidates", help="Required for fair baseline-vs-reranker comparison; checks unchanged boundaries")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = evaluate(read_jsonl(args.predictions), read_jsonl(args.ground_truth),
                       read_jsonl(args.reference_candidates) if args.reference_candidates else None)
    metrics["files_sha256"] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for name, path in vars(args).items() if name != "output" and path}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics["brief"], indent=2))
    print(json.dumps(metrics["candidate_diagnostics"], indent=2))


if __name__ == "__main__":
    main()
