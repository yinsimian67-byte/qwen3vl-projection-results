"""CPU logic tests; real processor tests are opt-in and never load model weights."""
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

from vlm.common import (assistant_labels, checked_windows, join_inputs, max_tiou,
                        pairwise_order, pointwise_order, prompt_messages,
                        ranked_prediction, tiou)
from vlm.prepare_sft import make_examples


class VLMLogicTests(unittest.TestCase):
    def setUp(self):
        self.annotation = {"qid": 7, "vid": "a_0_12", "query": "A person runs.",
                           "duration": 12, "relevant_windows": [[4, 8]], "split": "train"}
        self.candidate = {"qid": 7, "vid": "a_0_12", "split": "train",
                          "pred_relevant_windows": [[0, 2, 0.9], [4, 8, 0.7], [3, 5, 0.4]]}
        self.windows = checked_windows(self.candidate, 12)

    def test_iou_and_validation(self):
        self.assertAlmostEqual(tiou([0, 4], [2, 6]), 1 / 3)
        self.assertEqual(max_tiou([1, 2], []), 0)
        for invalid in [[0, 0, 1], [-1, 2, 1], [0, 13, 1], [0, 2, math.nan]]:
            with self.assertRaises(ValueError):
                checked_windows({"pred_relevant_windows": [invalid]}, 12)

    def test_exact_query_set_and_video_match(self):
        self.assertEqual(len(join_inputs([self.annotation], [self.candidate])), 1)
        with self.assertRaises(ValueError):
            join_inputs([self.annotation], [])
        with self.assertRaises(ValueError):
            join_inputs([self.annotation], [self.candidate, self.candidate])
        with self.assertRaises(ValueError):
            join_inputs([self.annotation], [{**self.candidate, "vid": "wrong"}])

    def test_course_preflight_matches_strict_evaluator(self):
        for windows in ([[0, 2, 0.9], [4, 8, 0.7], [11, 12.01, 0.4]],
                        [[0, 2, 0.1], [4, 8, 0.9], [8, 10, 0.4]],
                        [[0, 2, 0.9], [4, 8, 0.7]],
                        [[0, 2, 0.9], [4, 8, 0.7], [8, 10, True]]):
            with self.assertRaises(ValueError):
                join_inputs([self.annotation], [{**self.candidate, "pred_relevant_windows": windows}])
        with self.assertRaises(ValueError):
            join_inputs([{**self.annotation, "qid": "7"}], [self.candidate])
        with self.assertRaises(ValueError):
            join_inputs([], [])

    def test_fixed_pool_and_fallback(self):
        order, fallback = pointwise_order([0.2, 0.9, 0.4], 3)
        self.assertEqual(order, [1, 2, 0])
        self.assertFalse(fallback)
        prediction = ranked_prediction(self.annotation, self.windows, order)
        self.assertEqual(prediction["qid"], 7)
        self.assertEqual(prediction["pred_relevant_windows"][0][:2], [4.0, 8.0])
        self.assertEqual({tuple(w[:2]) for w in prediction["pred_relevant_windows"]},
                         {tuple(w[:2]) for w in self.windows})
        order, fallback = pointwise_order([0.9, None, 0.7], 3)
        self.assertTrue(fallback)
        self.assertEqual(order, [0, 1, 2])
        with self.assertRaises(ValueError):
            ranked_prediction(self.annotation, self.windows, [1, 1, 2])

    def test_pairwise_stable_order(self):
        strengths = [0.2, 0.8, 0.5]
        result = pairwise_order(3, 3, lambda left, right: 0.9 if strengths[left] > strengths[right] else 0.1)
        self.assertEqual(result, [1, 2, 0])
        self.assertEqual(pairwise_order(4, 3, lambda left, right: 0.5), [0, 1, 2, 3])

    def test_training_thresholds_no_gray_and_train_only(self):
        rows = make_examples(self.annotation, self.candidate, self.windows, "pointwise", random.Random(2026))
        self.assertEqual({row["answer"] for row in rows}, {"Yes", "No"})
        for row in rows:
            target = max_tiou(row["clips"][0], self.annotation["relevant_windows"])
            self.assertTrue(target >= 0.5 if row["answer"] == "Yes" else target <= 0.1)
        with self.assertRaises(ValueError):
            make_examples({**self.annotation, "split": "dev"}, self.candidate, self.windows, "pointwise", random.Random(0))
        with self.assertRaises(ValueError):
            make_examples(self.annotation, {**self.candidate, "split": "final"}, self.windows, "pointwise", random.Random(0))
        for missing_side in ("annotation", "candidate"):
            annotation, candidate = dict(self.annotation), dict(self.candidate)
            (annotation if missing_side == "annotation" else candidate).pop("split")
            with self.assertRaises(ValueError):
                make_examples(annotation, candidate, self.windows, "pointwise", random.Random(0))

    def test_multigt_training_label_matches_any_gt_not_first_only(self):
        annotation = {**self.annotation, "relevant_windows": [[0, 1], [4, 8]]}
        rows = make_examples(annotation, self.candidate, self.windows, "pointwise", random.Random(2026))
        for row in rows:
            expected = max_tiou(row["clips"][0], annotation["relevant_windows"])
            self.assertEqual(row["target_tiou"], expected)
            self.assertTrue(expected >= 0.5 if row["answer"] == "Yes" else expected <= 0.1)
        self.assertEqual(max_tiou([4, 8], annotation["relevant_windows"]), 1.0)

    def test_pairwise_training_positions_and_margin(self):
        answers = set()
        for seed in range(20):
            row = make_examples(self.annotation, self.candidate, self.windows, "pairwise", random.Random(seed))[0]
            answers.add(row["answer"])
            selected = 0 if row["answer"] == "A" else 1
            self.assertGreaterEqual(row["target_tiou"][selected] - row["target_tiou"][1 - selected], 0.4)
        self.assertEqual(answers, {"A", "B"})

    def test_answer_only_loss_mask(self):
        self.assertEqual(assistant_labels([11, 12, 90, 91], [11, 12]), [-100, -100, 90, 91])
        with self.assertRaises(ValueError):
            assistant_labels([11, 13, 90], [11, 12])
        with self.assertRaises(ValueError):
            assistant_labels([11, 12], [11, 12])

    def test_prompts_have_video_placeholders(self):
        for task, clips, expected in [("pointwise", [[0, 2]], 1), ("pairwise", [[0, 2], [4, 6]], 2)]:
            content = prompt_messages("run", clips, task)[0]["content"]
            self.assertEqual(sum(item["type"] == "video" for item in content), expected)
            self.assertNotIn("relevant_windows", str(content))


@unittest.skipUnless(importlib.util.find_spec("av") and importlib.util.find_spec("numpy"), "Requires PyAV and numpy")
class VideoCacheTests(unittest.TestCase):
    def test_decode_sampling_cache_and_missing_file(self):
        import av
        import numpy as np
        from vlm.video import VideoStore, decode_frames
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "clip.mp4"
            with av.open(str(path), "w") as output:
                stream = output.add_stream("mpeg4", rate=10)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                for index in range(40):
                    frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 5, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            store = VideoStore(root, root / "cache", 4, 64)
            frames, fps = store.get("clip", 1.0, 3.0)
            cached, cached_fps = store.get("clip", 1.0, 3.0)
            self.assertEqual(frames.shape, (4, 48, 64, 3))
            self.assertEqual(fps, 2.0)
            self.assertEqual(cached_fps, fps)
            np.testing.assert_array_equal(frames, cached)
            self.assertTrue(all(frames[i].mean() < frames[i + 1].mean() for i in range(3)))
            # Unsupported seeking must preserve exact samples and expose its
            # slower sequential fallback in audit metadata.
            real_open = av.open
            opens = []
            class NoSeekContainer:
                def __init__(self, container):
                    self.container = container
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    self.container.close()
                def __getattr__(self, name):
                    return getattr(self.container, name)
                def seek(self, *args, **kwargs):
                    raise RuntimeError("synthetic unsupported seek")
            def injected_open(*args, **kwargs):
                container = real_open(*args, **kwargs)
                opens.append(container)
                return NoSeekContainer(container) if len(opens) == 1 else container
            with patch("av.open", side_effect=injected_open):
                fallback_frames, _, fallback_fps, fallback_sampling = decode_frames(path, 1, 3, 4, 64)
            np.testing.assert_array_equal(fallback_frames, frames)
            self.assertEqual(fallback_fps, fps)
            self.assertEqual(fallback_sampling["decode_strategy"], "sequential_fallback")
            self.assertIn("unsupported seek", fallback_sampling["seek_fallback"])
            with self.assertRaises(FileNotFoundError):
                store.get("missing", 0, 2)
            with self.assertRaises(ValueError):
                store.get("clip", 1, 8)
            tail_frames, tail_fps = store.get("clip", 3, 5)
            self.assertEqual(tail_frames.shape[0], 4)
            self.assertAlmostEqual(tail_fps, 4.0)
            adjustment = store.drain_sampling_events()[0]
            self.assertAlmostEqual(adjustment["end_trim_seconds"], 1.0)
            self.assertEqual(adjustment["effective_end"], 4.0)
            with self.assertRaises(ValueError):
                store.get("clip", 4.1, 5)
            # The full cache can move to a different machine without any source
            # video, absolute-path dependency, or matching modification times.
            relocated = root / "elsewhere" / "portable_cache"
            shutil.copytree(root / "cache", relocated)
            path.unlink()
            portable = VideoStore(root / "no_videos", relocated, 4, 64, cache_only=True)
            moved, moved_fps = portable.get("clip", 1, 3)
            np.testing.assert_array_equal(moved, frames)
            self.assertEqual(moved_fps, fps)
            self.assertEqual(len(portable.manifest), 2)
            self.assertEqual(len(next(iter(portable.manifest.values()))["source_sha256"]), 64)
            with self.assertRaises(FileNotFoundError):
                portable.get("clip", 0, 2)
            _, moved_tail_fps = portable.get("clip", 3, 5)
            self.assertEqual(moved_tail_fps, 4.0)
            self.assertAlmostEqual(portable.drain_sampling_events()[0]["end_trim_seconds"], 1.0)


@unittest.skipUnless(os.environ.get("QVH_PROCESSOR_TEST") == "1" or os.environ.get("QVH_PROCESSOR_PATH"), "Set QVH_PROCESSOR_TEST=1 or QVH_PROCESSOR_PATH to load the official processor/tokenizer")
class RealProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor
        local_processor = os.environ.get("QVH_PROCESSOR_PATH")
        cls.processor = AutoProcessor.from_pretrained(
            local_processor or "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=3136, max_pixels=50176,
            revision="66285546d2b821cf421d4f5eb2576359d3770cd3", use_fast=False,
            local_files_only=bool(local_processor),
        )

    def test_real_pointwise_and_pairwise_mask_and_time(self):
        import numpy as np
        from vlm.modeling import prepare_inputs
        class SyntheticStore:
            def get(self, vid, start, end):
                return np.zeros((4, 56, 56, 3), dtype=np.uint8), 4 / (end - start)
        for task, clips, answer in [("pointwise", [[0, 8]], "Yes"), ("pairwise", [[0, 8], [2, 6]], "B")]:
            batch = prepare_inputs(self.processor, "A person runs.", clips, task, SyntheticStore(), "synthetic", answer)
            inference = prepare_inputs(self.processor, "A person runs.", clips, task, SyntheticStore(), "synthetic")
            length = inference["input_ids"].shape[1]
            self.assertTrue((batch["labels"][0, :length] == -100).all())
            self.assertTrue((batch["labels"][0, length:] != -100).all())
            self.assertEqual(batch["labels"][0, length].item(), self.processor.tokenizer.encode(answer, add_special_tokens=False)[0])
            self.assertEqual(batch["video_grid_thw"].shape[0], len(clips))
            expected_seconds = [(end - start) / 2 for start, end in clips]
            self.assertEqual(list(batch["second_per_grid_ts"]), expected_seconds)
            text = self.processor.tokenizer.decode(batch["labels"][0, length:].tolist())
            self.assertTrue(text.startswith(answer + "<|im_end|>"), text)

    def test_tiny_random_qwen_forward_and_lora_gradient(self):
        """Check the actual architecture API on CPU; NOT a 3B GPU benchmark."""
        import numpy as np
        import torch
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model
        from vlm.modeling import prepare_inputs
        torch.set_num_threads(2)
        tokenizer = self.processor.tokenizer
        config = Qwen2_5_VLConfig(
            vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                           "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
                           "out_hidden_size": 64, "window_size": 112, "fullatt_block_indexes": [0]},
            video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
            image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        )
        config._attn_implementation = "eager"
        model = Qwen2_5_VLForConditionalGeneration(config)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        targets = [name for name, _ in model.named_modules()
                   if ".self_attn." in name and "visual" not in name and name.endswith((".q_proj", ".v_proj"))]
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=targets))
        class SyntheticStore:
            def get(self, vid, start, end):
                return np.zeros((4, 56, 56, 3), dtype=np.uint8), 4 / (end - start)
        batch = prepare_inputs(self.processor, "A person runs.", [[0, 8], [2, 6]], "pairwise", SyntheticStore(), "synthetic", "B")
        loss = model(**batch, use_cache=False).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [parameter.grad for name, parameter in model.named_parameters() if "lora_B" in name]
        self.assertTrue(all(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients))
        self.assertFalse(any(parameter.requires_grad for name, parameter in model.named_parameters() if "visual" in name))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires torch, but no GPU")
class InferenceAuditTests(unittest.TestCase):
    def test_parameter_update_evidence_tracks_actual_adamw_changes(self):
        import torch
        from vlm.train import snapshot_trainable, update_evidence
        model = torch.nn.Module()
        model.register_parameter("frozen_backbone", torch.nn.Parameter(torch.ones(1024), requires_grad=False))
        model.register_parameter("lora_A", torch.nn.Parameter(torch.tensor([[1.0, 2.0]])))
        optimizer = torch.optim.AdamW([model.lora_A], lr=0.01)
        initial = snapshot_trainable(model)
        self.assertEqual(set(initial), {"lora_A"})
        before = update_evidence(model, initial, optimizer, 0, 0)
        self.assertEqual(before["total_delta_l2"], 0)
        self.assertFalse(before["evidence_of_parameter_update"])
        model.lora_A.square().sum().backward()
        norm = torch.nn.utils.clip_grad_norm_([model.lora_A], 1.0, error_if_nonfinite=True)
        self.assertGreater(float(norm), 0)
        optimizer.step()
        after = update_evidence(model, initial, optimizer, 0, 1)
        self.assertGreater(after["total_delta_l2"], 0)
        self.assertEqual(after["changed_elements"], 2)
        self.assertEqual(after["optimizer_state_step_min"], 1)
        self.assertNotEqual(after["initial_trainable_sha256"], after["final_trainable_sha256"])
        self.assertTrue(after["evidence_of_parameter_update"])
        from vlm.modeling import lora_parameter_digest
        model.lora_A.requires_grad_(False)
        self.assertEqual(lora_parameter_digest(model)["sha256"], after["final_trainable_sha256"])
        model.lora_A.requires_grad_(True)
        resumed_initial = snapshot_trainable(model)
        resumed = update_evidence(model, resumed_initial, optimizer, 1, 1)
        self.assertFalse(resumed["evidence_of_parameter_update"])

    def test_nonfinite_preference_is_explicit_failure(self):
        import torch
        from types import SimpleNamespace
        from vlm.modeling import label_probability
        class FakeTokenizer:
            def encode(self, text, **kwargs):
                return [0 if text == "Yes" else 1]
        def model(**kwargs):
            return SimpleNamespace(logits=torch.tensor([[[float("nan"), 0.0]]]))
        with patch("vlm.modeling.to_cuda", side_effect=lambda batch: batch):
            with self.assertRaises(FloatingPointError):
                label_probability(model, SimpleNamespace(tokenizer=FakeTokenizer()), {}, "pointwise")

    def test_rerank_cli_preserves_all_queries_and_audits_failure(self):
        from types import SimpleNamespace
        from vlm import rerank
        from vlm.common import write_jsonl, load_jsonl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = [{"qid": index, "vid": vid, "query": "A person runs.", "duration": 12}
                           for index, vid in [(1, "ok"), (2, "missing")]]
            candidates = [{"qid": row["qid"], "vid": row["vid"], "pred_relevant_windows": [[0, 2, 1], [4, 6, 0.8], [8, 10, 0.5]]}
                          for row in annotations]
            write_jsonl(root / "queries.jsonl", annotations)
            write_jsonl(root / "candidates.jsonl", candidates)
            config_path = Path(__file__).resolve().parents[1] / "configs" / "vlm_qlora_pointwise.json"
            argv = ["rerank", "--annotations", str(root / "queries.jsonl"), "--candidates", str(root / "candidates.jsonl"),
                    "--config", str(config_path), "--cache-only", "--cache-dir", str(root / "cache"), "--output", str(root / "out.jsonl")]
            def prepare(processor, query, clips, task, store, vid, **kwargs):
                if vid == "missing":
                    raise FileNotFoundError("Cache-only miss")
                return {"score": clips[0][0] / 10}
            with patch("sys.argv", argv), patch("vlm.rerank.load_model", return_value=(SimpleNamespace(config=SimpleNamespace()), None, None)), \
                 patch("vlm.rerank.VideoStore", return_value=SimpleNamespace(drain_sampling_events=lambda: [])), patch("vlm.rerank.prepare_inputs", side_effect=prepare), \
                 patch("vlm.rerank.label_probability", side_effect=lambda model, processor, batch, task: batch["score"]), \
                 patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=0), \
                 patch("torch.cuda.max_memory_reserved", return_value=0), patch("torch.cuda.get_device_name", return_value="mock GPU for CPU logic test"), patch("builtins.print"):
                rerank.main()
            predictions = load_jsonl(root / "out.jsonl")
            self.assertEqual([row["qid"] for row in predictions], [1, 2])
            self.assertEqual(predictions[0]["pred_relevant_windows"][0][:2], [8, 10])
            self.assertEqual([row[:2] for row in predictions[1]["pred_relevant_windows"]], [[0, 2], [4, 6], [8, 10]])
            audit = load_jsonl(root / "out.jsonl.audit.jsonl")
            self.assertEqual(audit[1]["status"], "failed_baseline_order")
            metrics = json.loads((root / "out.jsonl.metrics.json").read_text())
            self.assertEqual(metrics["failed_queries"], 1)
            self.assertFalse(metrics["complete_vlm_evaluation"])
            from vlm.common import file_sha256
            self.assertEqual(metrics["prediction_sha256"], file_sha256(root / "out.jsonl"))


if __name__ == "__main__":
    unittest.main()
