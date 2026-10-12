"""QLoRA fine-tuning of Gemma-2 2B on Bangla math (cleaned version of finetune_w_2.py).

Only needed if you want to retrain; the trained adapter is already on the Hub
(ajoysr/bangla-math-gemma) and merge_lora.py downloads it.

    python src/train_lora.py                       # train and save locally
    HF_TOKEN=... python src/train_lora.py --push   # also push the adapter to the Hub

Changes from the original notebook:
- No access token in code. Use `huggingface-cli login` or the HF_TOKEN env variable.
- No explicit "<bos>" in the text: the tokenizer already adds one, so the original
  produced a double <bos> on every example.
- Training rows come from common.TRAIN_ROWS, so evaluation never overlaps training.
"""

import argparse
import os

from unsloth import FastLanguageModel, is_bfloat16_supported  # import unsloth before trl/transformers

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from common import ADAPTER_ID, DATASET_ID, TRAIN_ROWS, build_training_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--out", default="models/lora-adapter")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--hub-id", default=ADAPTER_ID)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-2-2b-it-bnb-4bit",
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    train = load_dataset(DATASET_ID, split="train").select(range(TRAIN_ROWS))
    train = train.map(
        lambda ex: {"text": [build_training_text(p, s) for p, s in zip(ex["problem"], ex["solution"])]},
        batched=True,
        remove_columns=train.column_names,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=30,
            max_steps=args.max_steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="checkpoints",
            save_steps=100,
            save_total_limit=2,
            report_to="none",
        ),
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved adapter to {args.out}")

    if args.push:
        token = os.environ.get("HF_TOKEN")  # None falls back to the `huggingface-cli login` token
        model.push_to_hub(args.hub_id, token=token)
        tokenizer.push_to_hub(args.hub_id, token=token)
        print(f"Pushed adapter to {args.hub_id}")


if __name__ == "__main__":
    main()
