"""Find the English NuminaMath-CoT original of each Bangla training problem.

The Bangla dataset is a translation of NuminaMath-CoT, but its rows are in a different order.
The translation keeps numbers and LaTeX unchanged, so a problem is identified by the sequence
of numbers in its statement together with its boxed answer. Only keys that are unique on both
sides are accepted.

    python src/match_english.py          -> results/english_parallel.json  {bangla_idx: english_idx}
"""

import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from common import BN_DIGITS, DATASET_ID, TRAIN_ROWS, last_boxed, normalize

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def key(problem, solution):
    numbers = tuple(NUMBER.findall(problem.translate(BN_DIGITS)))
    answer = last_boxed(solution)
    if len(numbers) < 2 or answer is None:
        return None  # too little to identify the problem reliably
    return numbers, normalize(answer)


def main():
    bangla = load_dataset(DATASET_ID, split="train").select(range(TRAIN_ROWS))
    english = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    bangla_keys = {i: key(r["problem"], r["solution"]) for i, r in enumerate(bangla)}
    wanted = Counter(k for k in bangla_keys.values() if k)

    english_index, english_counts = {}, Counter()
    for j, (p, s) in enumerate(zip(english["problem"], english["solution"])):
        numbers = tuple(NUMBER.findall(p))
        if numbers and len(numbers) >= 2:
            k = None
            answer = last_boxed(s)
            if answer is not None:
                k = (numbers, normalize(answer))
            if k in wanted:
                english_counts[k] += 1
                english_index[k] = j

    matches = {i: english_index[k] for i, k in bangla_keys.items()
               if k and wanted[k] == 1 and english_counts[k] == 1}
    Path("results").mkdir(exist_ok=True)
    Path("results/english_parallel.json").write_text(json.dumps(matches))
    print(f"matched {len(matches)} of {TRAIN_ROWS} Bangla training problems "
          f"({100 * len(matches) / TRAIN_ROWS:.0f}%)")

    for i in list(matches)[:3]:
        print("BN:", bangla[i]["problem"][:100].replace("\n", " "))
        print("EN:", english[matches[i]]["problem"][:100].replace("\n", " "), "\n")


if __name__ == "__main__":
    main()
