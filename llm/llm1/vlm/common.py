"""Data and ranking utilities; deliberately usable without torch or transformers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indexed(rows, name):
    if not rows:
        raise ValueError(f"Empty {name}")
    result = {}
    for row in rows:
        if not isinstance(row.get("qid"), int) or isinstance(row["qid"], bool):
            raise ValueError(f"qid must be an integer in {name}")
        key = str(row["qid"])
        if key in result:
            raise ValueError(f"Duplicate qid {key} in {name}")
        result[key] = row
    return result


def tiou(left, right):
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def max_tiou(window, ground_truth):
    return max((tiou(window, gt) for gt in ground_truth), default=0.0)


def checked_windows(row, duration):
    """Validate, without silently changing the fixed proposal pool or its order."""
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a finite positive number")
    windows = row.get("pred_relevant_windows", [])
    if not windows:
        raise ValueError(f"Empty candidate pool for qid {row.get('qid')}")
    result = []
    for window in windows:
        if not isinstance(window, (list, tuple)) or len(window) != 3:
            raise ValueError("Expected [start, end, score] for every candidate")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in window):
            raise ValueError("Candidate values must be numeric, not strings or booleans")
        start, end, score = map(float, window)
        if not all(math.isfinite(v) for v in (start, end, score)):
            raise ValueError("Nonfinite candidate value")
        if start < 0 or end <= start or end > float(duration):
            raise ValueError(f"Invalid candidate {window} for duration {duration}")
        result.append([start, end, score])
    return result


def join_inputs(annotations, candidates):
    ann_index = indexed(annotations, "annotations")
    cand_index = indexed(candidates, "candidates")
    missing = set(ann_index) - set(cand_index)
    extra = set(cand_index) - set(ann_index)
    if missing or extra:
        raise ValueError(f"Candidate/annotation qid sets differ: missing={len(missing)}, extra={len(extra)}")
    joined = []
    for annotation in annotations:
        candidate = cand_index[str(annotation["qid"])]
        if candidate.get("vid") != annotation["vid"]:
            raise ValueError(f"Video mismatch for qid {annotation['qid']}")
        windows = checked_windows(candidate, annotation["duration"])
        if len(windows) != 3:
            raise ValueError("The course protocol requires exactly three candidates per query")
        if any(windows[index][2] < windows[index+1][2] for index in range(len(windows)-1)):
            raise ValueError("Candidate scores must follow non-increasing input rank order")
        joined.append((annotation, candidate, windows))
    return joined


def ranked_prediction(annotation, windows, order):
    if sorted(order) != list(range(len(windows))):
        raise ValueError("Reranking must be a permutation of the complete candidate pool")
    # Official evaluation sorts by score. Ordinal scores guarantee the exact order;
    # probabilities and original scores are retained in the separate audit file.
    size = len(order)
    return {
        "qid": annotation["qid"], "vid": annotation["vid"],
        "query": annotation["query"],
        "pred_relevant_windows": [
            [windows[index][0], windows[index][1], (size - rank) / size]
            for rank, index in enumerate(order)
        ],
    }


def pointwise_order(scores, size):
    """One failed candidate reverts the whole query, avoiding failure-score bias."""
    if any(score is None for score in scores):
        return list(range(size)), True
    head = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return head + list(range(len(scores), size)), False


def pairwise_order(size, top_k, compare):
    """Stable insertion sort; compare(A,B) returns P(A preferred over B)."""
    order = []
    for candidate in range(min(top_k, size)):
        inserted = False
        for position, incumbent in enumerate(order):
            if compare(candidate, incumbent) > 0.5:
                order.insert(position, candidate)
                inserted = True
                break
        if not inserted:
            order.append(candidate)
    return order + list(range(min(top_k, size), size))


def prompt_messages(query, clips, task):
    if task == "pointwise":
        if len(clips) != 1:
            raise ValueError("Pointwise input must contain one candidate clip")
        content = [
            {"type": "video"},
            {"type": "text", "text": (
                f"Query: {query}\n"
                "Does this video segment give a good temporal match to the query, "
                "showing the described event with little unrelated content? "
                "Answer only Yes or No."
            )},
        ]
    elif task == "pairwise":
        if len(clips) != 2:
            raise ValueError("Pairwise input must contain two candidate clips")
        content = [
            {"type": "text", "text": "Candidate A:"}, {"type": "video"},
            {"type": "text", "text": "Candidate B:"}, {"type": "video"},
            {"type": "text", "text": (
                f"Query: {query}\n"
                "Which candidate is a better temporal match to the query, "
                "covering the described event with less unrelated content? "
                "Answer only A or B."
            )},
        ]
    else:
        raise ValueError(f"Unknown task {task}")
    return [{"role": "user", "content": content}]


def assistant_labels(full_ids, prefix_ids):
    """Mask all user/visual/header tokens; train only the answer and its EOS."""
    if len(full_ids) <= len(prefix_ids) or full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Chat prefix is not an exact token prefix; refusing an incorrect loss mask")
    return [-100] * len(prefix_ids) + list(full_ids[len(prefix_ids):])
