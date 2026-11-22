"""Pinned transformers 4.51.3 / PEFT 0.15.2 integration, with answer-only masks."""
from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

from .common import assistant_labels, prompt_messages


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["task"] not in {"pointwise", "pairwise"}:
        raise ValueError("Unknown VLM task")
    if config["num_frames"] < 2 or config["num_frames"] % 2:
        raise ValueError("num_frames must be even and >=2")
    return config


def load_model(config, training=False, adapter=None):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError("This VLM recipe needs a CUDA GPU. CPU tests do not load the 3B model.")
    dtype_name = config.get("dtype", "bfloat16")
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16; copy the config and set dtype=float16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    kwargs = {
        "torch_dtype": dtype, "device_map": {"": 0},
        "attn_implementation": "sdpa", "revision": config.get("revision", "main"),
    }
    if config.get("load_in_4bit", False):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["model_name"], **kwargs)
    processor = AutoProcessor.from_pretrained(
        config["model_name"], revision=config.get("revision", "main"),
        min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
        use_fast=False,
    )
    model.config.use_cache = False
    if training:
        if adapter:
            raise ValueError("Training resume is via --resume; do not mix adapter loading modes")
        if config.get("load_in_4bit", False):
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        suffixes = tuple("." + suffix for suffix in config["lora_targets"])
        targets = [name for name, _ in model.named_modules()
                   if ".self_attn." in name and "visual" not in name and name.endswith(suffixes)]
        if not targets:
            raise ValueError("Could not find language attention LoRA target modules")
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=config["lora_rank"],
            lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
            bias="none", target_modules=targets,
        ))
        # No vision encoder or language backbone weight is trainable.
        if any(parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()):
            raise RuntimeError("Unexpected trainable parameter outside LoRA")
        model.print_trainable_parameters()
    elif adapter:
        saved_config_path = Path(adapter) / "training_config.json"
        if saved_config_path.exists():
            saved = json.loads(saved_config_path.read_text())
            for key in ("model_name", "task", "num_frames", "min_pixels", "max_pixels"):
                if saved[key] != config[key]:
                    raise ValueError(f"Adapter/inference configuration mismatch: {key}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.train(training)
    return model, processor, dtype


def prepare_inputs(processor, query, clips, task, video_store, vid, answer=None, max_tokens=2048):
    import torch

    frames, frame_rates = [], []
    for start, end in clips:
        video, fps = video_store.get(vid, start, end)
        frames.append(video)
        frame_rates.append(fps)
    messages = prompt_messages(query, clips, task)
    prefix_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # In this pinned processor, fps builds second_per_grid_ts. Never drop this
    # field or substitute the original recording fps for the sampled-video fps.
    prefix = processor(text=[prefix_text], videos=frames, fps=frame_rates,
                       padding=False, return_tensors="pt")
    if "second_per_grid_ts" not in prefix or "video_grid_thw" not in prefix:
        raise RuntimeError("Processor did not produce Qwen video time-grid metadata")
    if answer is None:
        result = prefix
    else:
        full_text = processor.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False,
        )
        result = processor(text=[full_text], videos=frames, fps=frame_rates,
                           padding=False, return_tensors="pt")
        labels = assistant_labels(result["input_ids"][0].tolist(), prefix["input_ids"][0].tolist())
        result["labels"] = torch.tensor([labels], dtype=torch.long)
        result["labels"][result["attention_mask"] == 0] = -100
        # The same first-answer token is used by the inference preference score.
        answer_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
        first_answer = next(token for token in labels if token != -100)
        if len(answer_ids) != 1 or first_answer != answer_ids[0]:
            raise RuntimeError("Answer tokenization differs from the inference label token")
    if result["input_ids"].shape[1] > max_tokens:
        raise ValueError("Example exceeds max_tokens; reduce frames/pixels/query length rather than truncating visual tokens")
    return result


def to_cuda(batch):
    return {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in batch.items()}


def label_probability(model, processor, batch, task):
    import torch

    labels = ["Yes", "No"] if task == "pointwise" else ["A", "B"]
    encoded = [processor.tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(tokens) != 1 for tokens in encoded):
        raise RuntimeError("Pinned tokenizer must encode each answer as one token")
    with torch.inference_mode():
        logits = model(**to_cuda(batch), use_cache=False).logits[0, -1]
        probability = torch.softmax(logits[[tokens[0] for tokens in encoded]].float(), dim=-1)[0].item()
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise FloatingPointError("Model returned a nonfinite or invalid class probability")
    return probability


def lora_parameter_digest(model):
    """Hash adapter tensors only, including frozen adapters after inference load."""
    digest = hashlib.sha256()
    count, elements = 0, 0
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        value = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        count += 1
        elements += value.numel()
    return {"sha256": digest.hexdigest(), "tensor_count": count, "elements": elements}
