"""Merge the LoRA adapter into a 16-bit base model.

This 16-bit merged model is the reference ("no quantization") and the input to every
quantization method. Merging onto the 16-bit base (not the 4-bit one used during
training) avoids baking 4-bit rounding error into the baseline.

    python src/merge_lora.py --out models/merged-16bit
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ADAPTER_ID, BASE_MODEL_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE_MODEL_ID)
    parser.add_argument("--adapter", default=ADAPTER_ID)
    parser.add_argument("--out", default="models/merged-16bit")
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading base {args.base} in {dtype}")
    device_map = "auto" if torch.cuda.is_available() else None  # merge on CPU elsewhere (no disk offload)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print(f"Applying adapter {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"Saved merged model to {args.out}")


if __name__ == "__main__":
    main()
