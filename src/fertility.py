"""How many tokens does Gemma's tokenizer spend on Bangla compared with English?

Higher fertility (tokens per word) means longer sequences for the same content: more
activations per problem, more steps per solution, and more chances for quantization
error to build up. Writes results/fertility.json and paper/tables/fertility.tex.

    python src/fertility.py
"""

import json
import re
import statistics
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from common import BASE_MODEL_ID, DATASET_ID, TRAIN_ROWS

N = 1000
MATH = re.compile(r"\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S)


def stats(tokenizer, texts):
    words, tokens, chars = 0, 0, 0
    for t in texts:
        prose = MATH.sub(" ", t)  # count natural-language words only
        words += len(prose.split())
        tokens += len(tokenizer(prose, add_special_tokens=False)["input_ids"])
        chars += len(prose)
    return {"tokens_per_word": tokens / words, "chars_per_token": chars / tokens}


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    bangla = load_dataset(DATASET_ID, split="train").select(range(TRAIN_ROWS, TRAIN_ROWS + N))["solution"]
    english = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
               if len(t) > 300][:N]

    result = {
        "bangla_math_solutions": stats(tokenizer, bangla),
        "english_wikitext": stats(tokenizer, english),
        "bangla_solution_tokens_median": statistics.median(
            len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in bangla),
    }
    result["fertility_ratio"] = (result["bangla_math_solutions"]["tokens_per_word"]
                                 / result["english_wikitext"]["tokens_per_word"])

    Path("results").mkdir(exist_ok=True)
    Path("results/fertility.json").write_text(json.dumps(result, indent=2))

    b, e = result["bangla_math_solutions"], result["english_wikitext"]
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    Path("paper/tables/fertility.tex").write_text(
        f"\\newcommand{{\\FertBangla}}{{{b['tokens_per_word']:.2f}}}\n"
        f"\\newcommand{{\\FertEnglish}}{{{e['tokens_per_word']:.2f}}}\n"
        f"\\newcommand{{\\FertRatio}}{{{result['fertility_ratio']:.1f}}}\n"
        f"\\newcommand{{\\SolutionTokens}}{{{result['bangla_solution_tokens_median']:.0f}}}\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
