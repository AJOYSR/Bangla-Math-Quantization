"""Quantize the merged model in memory (simulated quantization) and evaluate it.

Runs on Apple Silicon (MPS), CUDA or CPU. See simquant.py for what "simulated" means.

    python src/simulate.py --method 16bit
    python src/simulate.py --method rtn  --bits 8 --group-size -1
    python src/simulate.py --method nf4
    python src/simulate.py --method rtn  --bits 4
    python src/simulate.py --method gptq --bits 4 --calib bangla --seed 0

Writes results/<variant>.json and results/<variant>.jsonl. An interrupted run resumes from
results/<variant>.partial.jsonl.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import is_correct, last_boxed, load_calibration_texts, load_eval_set
from evaluate import generate
from simquant import apply_gptq, apply_nf4, apply_rtn, bits_per_weight, model_size_gb


def variant_name(args):
    if args.method == "16bit":
        return "16bit" + (f"-b{args.batch_size}" if args.batch_size != 16 else "")
    if args.method == "nf4":
        return "nf4"
    if args.method == "rtn":
        return f"rtn-w{args.bits}"
    return f"gptq-w{args.bits}-{args.calib}-s{args.seed}"


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["16bit", "rtn", "nf4", "gptq"])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128, help="-1 = one scale per output row")
    parser.add_argument("--calib", default="bangla", choices=["bangla", "english-math", "english", "bangla-parallel", "english-parallel"])
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--path", default="models/merged-16bit")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-ref-tokens", type=int, default=384,
                        help="evaluate only problems whose reference solution fits in this many tokens")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    if args.method == "nf4":
        args.bits, args.group_size = 4, 64
    if args.method == "16bit":
        args.bits = 16

    name = variant_name(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / f"{name}.json").exists():
        print(f"[{name}] already done, skipping")
        return

    device = pick_device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.path, torch_dtype=dtype, attn_implementation="eager").to(device)
    model.eval()

    t0 = time.time()
    if args.method == "rtn":
        apply_rtn(model, args.bits, args.group_size)
    elif args.method == "nf4":
        apply_nf4(model)
    elif args.method == "gptq":
        texts = load_calibration_texts(args.calib, n=args.n_calib, seed=args.seed)
        apply_gptq(model, tokenizer, texts, args.bits, args.group_size, log=lambda m: print(f"[{name}]{m}", flush=True))
    quant_minutes = (time.time() - t0) / 60
    size_gb, n_quantized, n_total = model_size_gb(model, args.method, args.bits, args.group_size)
    print(f"[{name}] quantized in {quant_minutes:.1f} min, size {size_gb:.2f} GB", flush=True)

    items = load_eval_set(args.n, max_ref_tokens=args.max_ref_tokens)
    # Batch problems with similar expected output length, so short ones don't wait for long ones.
    items.sort(key=lambda it: (it["ref_tokens"] or 0, len(it["problem"])))

    partial = out_dir / f"{name}.partial.jsonl"
    records = [json.loads(l) for l in partial.read_text(encoding="utf-8").splitlines()] if partial.exists() else []
    done = {r["idx"] for r in records}
    todo = [it for it in items if it["idx"] not in done]
    if done:
        print(f"[{name}] resuming: {len(done)} done, {len(todo)} left", flush=True)

    t0, gen_tokens = time.time(), 0
    with open(partial, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            texts, n_tokens = generate(model, tokenizer, [it["problem"] for it in batch], args.max_new_tokens)
            gen_tokens += sum(n_tokens)
            for it, text, n in zip(batch, texts, n_tokens):
                r = {
                    "idx": it["idx"],
                    "gold": it["answer"],
                    "pred": last_boxed(text),
                    "correct": is_correct(text, it["answer"]),
                    "n_tokens": n,
                    "hit_max_tokens": n >= args.max_new_tokens,
                    "output": text,
                }
                records.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"[{name}] {len(records)}/{len(items)}  accuracy so far {acc:.1%}", flush=True)
    eval_minutes = (time.time() - t0) / 60

    summary = {
        "variant": name,
        "simulated": True,
        "method": args.method,
        "bits": args.bits,
        "group_size": args.group_size,
        "calib": args.calib if args.method == "gptq" else None,
        "n_calib": args.n_calib if args.method == "gptq" else None,
        "seed": args.seed if args.method == "gptq" else None,
        "n": len(records),
        "max_new_tokens": args.max_new_tokens,
        "max_ref_tokens": args.max_ref_tokens,
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "boxed_rate": sum(r["pred"] is not None for r in records) / len(records),
        "bits_per_weight": round(bits_per_weight(args.method, args.bits, args.group_size), 3),
        "size_gb": round(size_gb, 3),
        "quantized_params": n_quantized,
        "total_params": n_total,
        "quant_minutes": round(quant_minutes, 2),
        "eval_minutes": round(eval_minutes, 2),
        "throughput_tok_s": round(gen_tokens / max(eval_minutes * 60, 1e-9), 1) if gen_tokens else None,
        "device": str(device),
    }
    with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(records, key=lambda r: r["idx"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
    partial.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
