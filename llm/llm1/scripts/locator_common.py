"""Shared course adapters around the official Moment-DETR model and dataset."""
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "moment_detr"
sys.path.insert(0, str(UPSTREAM))


def discover_features(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Feature root does not exist: {root}. Download and extract the feature archive first.")
    names = {"slowfast_features": ("slowfast_features", "slowfast"),
             "clip_features": ("clip_features", "clip"),
             "clip_text_features": ("clip_text_features", "clip_text")}
    found = {name: set() for name in names}
    for directory, children, _ in os.walk(root):
        path = Path(directory)
        for canonical, aliases in names.items():
            for candidate in [path] + [path / child for child in children]:
                if candidate.name in aliases and candidate.is_dir():
                    found[canonical].add(candidate.resolve())
        children[:] = [name for name in children if not name.startswith(".") and
                       not any(name in aliases for aliases in names.values())]
    for name, paths in found.items():
        if len(paths) != 1:
            raise ValueError(f"Expected one {name} directory below {root}; found {paths}. Point --feature-root to one extracted package.")
    return {name: str(next(iter(paths))) for name, paths in found.items()}


def feature_file(directory, basename, text=False):
    path = Path(directory) / f"{basename}.npz"
    if path.is_file():
        return path
    if not text:
        path = Path(directory) / f"{basename}.npy"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing feature: {directory}/{basename}.{'npz' if text else '(npz|npy)'}")


def audit_features(layout, rows):
    import numpy as np
    if not rows:
        raise ValueError("No annotation rows")
    dimensions = {"slowfast_features": 2304, "clip_features": 512, "clip_text_features": 512}
    report = {"directories": layout, "checked_queries": len(rows), "samples": {}}
    for name, dimension in dimensions.items():
        text = name == "clip_text_features"
        basenames = sorted({f"qid{row['qid']}" if text else row["vid"] for row in rows})
        paths = [feature_file(layout[name], basename, text=text) for basename in basenames]
        sample = paths[0]
        if sample.suffix == ".npz":
            with np.load(sample, allow_pickle=False) as archive:
                array = archive["last_hidden_state" if text else "features"]
        else:
            array = np.load(sample, allow_pickle=False, mmap_mode="r")
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[-1] != dimension:
            raise ValueError(f"Unexpected feature shape {array.shape} in {sample}; expected (T,{dimension})")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite features in {sample}")
        report["samples"][name] = {"file": str(sample), "shape": list(array.shape), "files_present": len(paths)}
    return report


def model_config(device="cpu"):
    from moment_detr.config import BaseOptions
    options = BaseOptions()
    options.initialize()
    config = vars(options.parser.parse_args([]))
    config.update(device=device, dset_name="hl", ctx_mode="video_tef", v_feat_dim=2818,
                  t_feat_dim=512, max_q_l=32, max_v_l=75, clip_length=2,
                  max_windows=20, span_loss_type="l1", num_queries=10)
    return config


def make_dataset(annotations, layout, config, load_labels=False):
    from moment_detr.start_end_dataset import StartEndDataset
    return StartEndDataset(dset_name="hl", data_path=str(annotations),
                           v_feat_dirs=[layout["slowfast_features"], layout["clip_features"]],
                           q_feat_dir=layout["clip_text_features"], q_feat_type="last_hidden_state",
                           max_q_l=config["max_q_l"], max_v_l=config["max_v_l"],
                           ctx_mode="video_tef", normalize_v=True, normalize_t=True,
                           load_labels=load_labels, clip_len=2, max_windows=config["max_windows"],
                           span_loss_type="l1", txt_drop_ratio=0)


def sanitize_window(start, end, duration):
    """Round to the 2 s feature grid, clamp to this clip, retain nonempty windows."""
    if not all(math.isfinite(x) for x in (start, end, duration)) or duration <= 0:
        raise ValueError("Invalid predicted span/duration")
    start = min(max(round(start / 2) * 2, 0.0), max(duration - 2, 0.0))
    end = min(max(round(end / 2) * 2, start + min(2, duration)), duration)
    return float(start), float(end)


def predict(model, loader, device, split):
    import torch
    from moment_detr.start_end_dataset import prepare_batch_inputs
    from moment_detr.span_utils import span_cxw_to_xx
    model.eval()
    predictions = []
    with torch.inference_mode():
        for metadata, batch in loader:
            inputs, _ = prepare_batch_inputs(batch, device, non_blocking=device.type == "cuda")
            output = model(**inputs)
            scores = output["pred_logits"].softmax(-1)[..., 0].cpu()
            spans = span_cxw_to_xx(output["pred_spans"]).cpu()
            for row, row_spans, row_scores in zip(metadata, spans, scores):
                ranked = sorted(zip(row_spans.tolist(), row_scores.tolist()), key=lambda item: item[1], reverse=True)[:3]
                windows = []
                for span, score in ranked:
                    start, end = sanitize_window(span[0] * row["duration"], span[1] * row["duration"], row["duration"])
                    windows.append([start, end, round(score, 8)])
                predictions.append({"qid": row["qid"], "vid": row["vid"], "query": row["query"],
                                    "duration": row["duration"], "split": split, "pred_relevant_windows": windows})
    return predictions


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
