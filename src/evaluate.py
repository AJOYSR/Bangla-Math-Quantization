"""Evaluate one model variant: accuracy, memory, size on disk, latency and throughput.

    python src/evaluate.py --variant 16bit
    python src/evaluate.py --variant bnb-int8
    python src/evaluate.py --variant bnb-nf4
    python src/evaluate.py --variant gptq-w4-bangla-s0 --path models/gptq-w4-bangla-s0

Writes results/<variant>.json (summary) and results/<variant>.jsonl (per-problem outputs).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import build_prompt, is_correct, last_boxed, load_eval_set

MERGED = "models/merged-16bit"


def compute_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_variant(variant: str, path: str):
    """Return (model, tokenizer, size_on_disk_gb or None)."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    common_kwargs = dict(device_map="cuda", attn_implementation="eager")  # eager is recommended for Gemma-2

    if variant == "16bit":
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=compute_dtype(), **common_kwargs)
    elif variant == "bnb-int8":
        quant = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=quant, **common_kwargs)
    elif variant == "bnb-nf4":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
        )
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=quant, **common_kwargs)
    elif variant.startswith("gptq"):
        from gptqmodel import GPTQModel
        model = GPTQModel.load(path, device="cuda").model
    else:
        raise ValueError(f"unknown variant: {variant}")

    # bitsandbytes quantizes on load, so only 16-bit and GPTQ have a meaningful file size.
    size_gb = None
    if variant == "16bit" or variant.startswith("gptq"):
        size_gb = sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1024**3

    model.eval()
    return model, tokenizer, size_gb


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def stop_token_ids(tokenizer):
    return [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]


@torch.no_grad()
def generate(model, tokenizer, problems, max_new_tokens):
    prompts = [build_prompt(p) for p in problems]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy: deterministic, so variants are directly comparable
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_token_ids(tokenizer),
    )
    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    per_row = (new_tokens != tokenizer.pad_token_id).sum(dim=1).tolist()
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return texts, per_row


@torch.no_grad()
def single_request_latency(model, tokenizer, problem, n_tokens=128, repeats=5):
    """Median seconds to produce n_tokens for one request (batch size 1)."""
    inputs = tokenizer([build_prompt(problem)], return_tensors="pt").to(model.device)
    times = []
    for _ in range(repeats + 1):  # first run is warm-up
        synchronize(model.device)
        start = time.perf_counter()
        model.generate(**inputs, max_new_tokens=n_tokens, min_new_tokens=n_tokens,
                       do_sample=False, pad_token_id=tokenizer.pad_token_id)
        synchronize(model.device)
        times.append(time.perf_counter() - start)
    return statistics.median(times[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--path", default=MERGED)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_eval_set(args.n)
    # Group similar-length prompts so batches waste less padding.
    items.sort(key=lambda it: len(it["problem"]))

    model, tokenizer, size_gb = load_variant(args.variant, args.path)
    tokenizer.padding_side = "left"
    weights_gb = model.get_memory_footprint() / 1024**3
    print(f"[{args.variant}] weights in GPU memory: {weights_gb:.2f} GB")

    latency = single_request_latency(model, tokenizer, items[0]["problem"])
    print(f"[{args.variant}] batch-1 latency for 128 tokens: {latency:.2f} s")

    torch.cuda.reset_peak_memory_stats()
    records, total_tokens, total_time = [], 0, 0.0
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        t0 = time.perf_counter()
        texts, n_tokens = generate(model, tokenizer, [it["problem"] for it in batch], args.max_new_tokens)
        total_time += time.perf_counter() - t0
        total_tokens += sum(n_tokens)

        for it, text, n in zip(batch, texts, n_tokens):
            records.append({
                "idx": it["idx"],
                "gold": it["answer"],
                "pred": last_boxed(text),
                "correct": is_correct(text, it["answer"]),
                "n_tokens": n,
                "hit_max_tokens": n >= args.max_new_tokens,
                "output": text,
            })
        done = len(records)
        acc = sum(r["correct"] for r in records) / done
        print(f"[{args.variant}] {done}/{len(items)}  running accuracy {acc:.1%}")

    summary = {
        "variant": args.variant,
        "n": len(records),
        "max_new_tokens": args.max_new_tokens,
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "boxed_rate": sum(r["pred"] is not None for r in records) / len(records),
        "weights_gb": round(weights_gb, 3),
        "size_on_disk_gb": round(size_gb, 3) if size_gb is not None else None,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "latency_128_tokens_s": round(latency, 3),
        "throughput_tok_s": round(total_tokens / total_time, 1),
        "gpu": torch.cuda.get_device_name(),
    }

    with open(out_dir / f"{args.variant}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / f"{args.variant}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
