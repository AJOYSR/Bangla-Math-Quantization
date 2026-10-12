"""Shared helpers: prompt format, data splits, answer extraction and scoring."""

import random
import re

DATASET_ID = "hamim-87/Ashrafur_bangla_math"
BASE_MODEL_ID = "unsloth/gemma-2-2b-it"        # 16-bit counterpart of the 4-bit base used in training
ADAPTER_ID = "ajoysr/bangla-math-gemma"         # LoRA adapter pushed by train_lora.py
TRAIN_ROWS = 3000                               # rows [0, 3000) were used for LoRA training

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


# ----------------------------------------------------------------- prompts

def build_prompt(problem: str) -> str:
    """Gemma-2 chat prompt. The tokenizer adds <bos> itself."""
    return f"<start_of_turn>user\n{problem}<end_of_turn>\n<start_of_turn>model\n"


def build_training_text(problem: str, solution: str) -> str:
    return build_prompt(problem) + f"{solution}<end_of_turn><eos>"


# ----------------------------------------------------------------- data

def load_eval_set(n: int = 200, seed: int = 0, max_ref_tokens: int = None):
    """Held-out problems (never seen in training) with an auto-gradable \\boxed{} gold answer.

    max_ref_tokens keeps only problems whose reference solution is at most this many tokens,
    so that a fixed generation budget can fit a complete solution.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train")
    tokenizer = None
    if max_ref_tokens:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    candidates = list(range(TRAIN_ROWS, len(ds)))
    random.Random(seed).shuffle(candidates)

    items = []
    for idx in candidates:
        row = ds[idx]
        gold = last_boxed(row["solution"])
        if gold is None or not is_gradable(gold):
            continue
        ref_tokens = len(tokenizer(row["solution"], add_special_tokens=False).input_ids) if tokenizer else None
        if ref_tokens is not None and ref_tokens > max_ref_tokens:
            continue
        items.append({"idx": idx, "problem": row["problem"], "answer": gold, "ref_tokens": ref_tokens})
        if len(items) == n:
            break
    return items


def load_calibration_texts(lang: str, n: int = 256, seed: int = 0, max_chars: int = 2000):
    """Calibration samples for GPTQ.

    bangla:       in-domain Bangla math (from the training rows, so no test leakage)
    bangla-parallel / english-parallel: the same matched problems in Bangla and in the
                  original English (NuminaMath-CoT), so only the language differs
    english-math: English grade-school math (GSM8K), a different source
    english:      generic English Wikipedia text (the usual default in GPTQ papers)
    """
    from datasets import load_dataset

    if lang == "bangla":
        ds = load_dataset(DATASET_ID, split="train").select(range(TRAIN_ROWS))
        rows = random.Random(seed).sample(range(len(ds)), n)
        texts = [build_training_text(ds[i]["problem"], ds[i]["solution"]) for i in rows]
    elif lang in ("bangla-parallel", "english-parallel"):
        # The same problems in both languages: Bangla training rows and their English
        # NuminaMath-CoT originals (matched by src/match_english.py). No character cut:
        # both languages are truncated at the same token budget during calibration.
        import json
        from pathlib import Path
        pairs = {int(k): v for k, v in json.loads(Path("results/english_parallel.json").read_text()).items()}
        rows = random.Random(seed).sample(sorted(pairs), n)
        if lang == "bangla-parallel":
            ds = load_dataset(DATASET_ID, split="train")
            return [build_training_text(ds[i]["problem"], ds[i]["solution"]) for i in rows]
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
        return [build_training_text(ds[pairs[i]]["problem"], ds[pairs[i]]["solution"]) for i in rows]
    elif lang == "english-math":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        rows = random.Random(seed).sample(range(len(ds)), n)
        texts = [build_training_text(ds[i]["question"], ds[i]["answer"]) for i in rows]
    elif lang == "english":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        paragraphs = [t for t in ds["text"] if len(t) > 500]
        texts = random.Random(seed).sample(paragraphs, n)
    else:
        raise ValueError(f"unknown calibration language: {lang}")

    return [t[:max_chars] for t in texts]


# ----------------------------------------------------------------- scoring

def last_boxed(text: str):
    """Content of the last \\boxed{...} in text, handling nested braces."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    open_idx = text.find("{", start)
    if open_idx < 0:
        return None
    depth = 0
    for j in range(open_idx, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:j]
    return None


BENGALI_LETTERS = re.compile(r"[\u0980-\u09FF]")


def normalize(ans: str) -> str:
    s = ans.translate(BN_DIGITS)
    s = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    for a, b in [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\$", ""), ("$", ""), (" ", "")]:
        s = s.replace(a, b)
    s = s.rstrip(".")
    # "1280বর্গমিটার" -> "1280": drop a Bangla unit word after a number
    unit = re.fullmatch(r"(-?[\d.,]+)[\u0980-\u09FF]+", s)
    if unit:
        s = unit.group(1)
    choice = re.fullmatch(r"\(?([A-Ea-e])\)?", s)
    if choice:
        return choice.group(1).upper()
    return s


def to_number(s: str):
    s = s.replace(",", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    slash = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if slash and float(slash.group(2)) != 0:
        return float(slash.group(1)) / float(slash.group(2))
    frac = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if frac and float(frac.group(3)) != 0:
        sign = -1.0 if frac.group(1) else 1.0
        return sign * float(frac.group(2)) / float(frac.group(3))
    return None


def answer_forms(normalized: str) -> set:
    """An AMC-style answer "(B)60" counts as "(B)60", "B" or "60"."""
    forms = {normalized}
    labeled = re.fullmatch(r"\(([A-E])\)(.+)", normalized)
    if labeled:
        forms |= {labeled.group(1), labeled.group(2)}
    return forms


def is_gradable(gold_answer: str) -> bool:
    """Free-text answers ("না, তারা ফিরে যেতে পারবে না") can't be scored by string match."""
    return not BENGALI_LETTERS.search(normalize(gold_answer))


def same_answer(p: str, g: str) -> bool:
    if p == g:
        return True
    pn, gn = to_number(p), to_number(g)
    return pn is not None and gn is not None and abs(pn - gn) <= 1e-6 * max(1.0, abs(gn))


OPTION_LINE = re.compile(r"^\s*\(?([A-E])[\):.]\s*(.+?)\s*$", re.M)


def choice_options(problem: str) -> dict:
    """{"A": "243", ...} for multiple-choice problems written as "A: 243" lines."""
    return {letter: normalize(text) for letter, text in OPTION_LINE.findall(problem)}


def is_correct(prediction_text: str, gold_answer: str, problem: str = None) -> bool:
    """problem (optional) lets a choice letter and its option value count as the same answer."""
    pred = last_boxed(prediction_text)
    if pred is None:
        return False
    preds, golds = answer_forms(normalize(pred)), answer_forms(normalize(gold_answer))
    if problem:
        options = choice_options(problem)
        for forms in (preds, golds):
            forms |= {options[f] for f in list(forms) if f in options}
    return any(same_answer(p, g) for p in preds for g in golds)
