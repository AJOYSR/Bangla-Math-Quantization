"""Collect results into tables, statistics and figures for the paper.

    python src/report.py

Reads   results/<variant>.json and results/<variant>.jsonl   (from simulate.py or evaluate.py)
Writes  results/summary.md
        results/fig_tradeoff.{pdf,png}, fig_calibration.{pdf,png}, fig_errors.{pdf,png}
        paper/tables/{main,errors,calibration,setup,composition}.tex

Variants named "<name>-s<seed>" (e.g. gptq-w4-bangla-s1) are runs of the same method with
different calibration samples; they are grouped and reported as mean ± std.
"""

import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import DATASET_ID, is_correct, normalize, to_number  # noqa: E402

RESULTS = Path("results")
TABLES = Path("paper/tables")
REFERENCE = "16bit"
N_BOOTSTRAP = 10_000

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
REF_LINE = "#a3a29c"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order

ORDER = ["16bit", "rtn-w8", "bnb-int8", "nf4", "bnb-nf4", "rtn-w4", "gptq-w4-bangla",
         "gptq-w4-english-math", "gptq-w4-english", "gptq-w4-bangla-parallel", "gptq-w4-english-parallel",
         "rtn-w3", "gptq-w3-bangla"]
LABELS = {
    "16bit": "16-bit (reference)",
    "rtn-w8": "RTN 8-bit",
    "bnb-int8": "LLM.int8 8-bit",
    "nf4": "NF4 4-bit",
    "bnb-nf4": "NF4 4-bit (bnb)",
    "rtn-w4": "RTN 4-bit",
    "gptq-w4-bangla": "GPTQ 4-bit, Bangla math",
    "gptq-w4-english-math": "GPTQ 4-bit, English GSM8K",
    "gptq-w4-english": "GPTQ 4-bit, English wiki",
    "gptq-w4-bangla-parallel": "GPTQ 4-bit, Bangla parallel",
    "gptq-w4-english-parallel": "GPTQ 4-bit, English parallel",
    "rtn-w3": "RTN 3-bit",
    "gptq-w3-bangla": "GPTQ 3-bit, Bangla math",
}


# ----------------------------------------------------------------- loading

def method_of(variant):
    return re.sub(r"-s\d+$", "", variant)


def load():
    runs = defaultdict(list)  # method -> [(summary, records)]
    for path in sorted(RESULTS.glob("*.json")):
        summary = json.loads(path.read_text())
        if "variant" not in summary:  # e.g. fertility.json
            continue
        records_path = path.with_suffix(".jsonl")
        records = [json.loads(l) for l in records_path.read_text(encoding="utf-8").splitlines()] \
            if records_path.exists() else []
        runs[method_of(summary["variant"])].append((summary, records))
    rescore(runs)
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def rescore(runs):
    """Score every saved output again with the current rules, so all variants are scored identically."""
    from datasets import load_dataset

    idxs = sorted({r["idx"] for rs in runs.values() for _, records in rs for r in records})
    if not idxs:
        return
    ds = load_dataset(DATASET_ID, split="train")
    problems = dict(zip(idxs, ds.select(idxs)["problem"]))
    for rs in runs.values():
        for summary, records in rs:
            for r in records:
                r["correct"] = is_correct(r["output"], r["gold"], problems[r["idx"]])
            if records:
                summary["accuracy"] = sum(r["correct"] for r in records) / len(records)


# ----------------------------------------------------------------- statistics

def per_problem_accuracy(runs):
    """idx -> fraction of runs (seeds) that got the problem right."""
    hits = defaultdict(list)
    for _, records in runs:
        for r in records:
            hits[r["idx"]].append(float(r["correct"]))
    return {idx: sum(v) / len(v) for idx, v in hits.items()}


def paired_bootstrap(a, b, seed=0):
    """Mean difference (a - b) over shared problems, its 95% bootstrap CI (pp), and a two-sided p-value."""
    shared = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in shared]
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    # round away floating-point noise, so a bound of exactly 0 is not read as significant
    means = sorted(round(sum(rng.choices(diffs, k=n)) / n, 9) for _ in range(N_BOOTSTRAP))
    lo, hi = means[int(0.025 * N_BOOTSTRAP)], means[int(0.975 * N_BOOTSTRAP) - 1]
    p = min(1.0, 2 * min(sum(m <= 0 for m in means), sum(m >= 0 for m in means)) / N_BOOTSTRAP)
    return {"d": round(100 * sum(diffs) / n, 6), "lo": round(100 * lo, 6), "hi": round(100 * hi, 6), "p": p}


def holm(tests):
    """Holm-Bonferroni: add "p_adj" to each test dict in a family."""
    tests = [t for t in tests if t is not None]
    m = len(tests)
    running = 0.0
    for rank, t in enumerate(sorted(tests, key=lambda t: t["p"])):
        running = max(running, min(1.0, (m - rank) * t["p"]))
        t["p_adj"] = running
    return tests


def mean_std(values):
    if len(values) == 1:
        return values[0], None
    return statistics.mean(values), statistics.stdev(values)


# ----------------------------------------------------------------- error analysis

MATH = re.compile(r"\$[^$]*\$|\\\[.*?\\\]|\\[a-zA-Z]+", re.S)


def bangla_ratio(text):
    """Share of letters that are Bengali script, ignoring LaTeX math."""
    prose = MATH.sub(" ", text)
    bengali = len(re.findall(r"[ঀ-৿]", prose))
    latin = len(re.findall(r"[A-Za-z]", prose))
    return bengali / (bengali + latin) if bengali + latin else 1.0


def is_repetitive(text, n=8, times=3):
    """True if some sequence of n words occurs at least `times` times (a repetition loop)."""
    words = text.split()
    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return bool(counts) and max(counts.values()) >= times


def error_breakdown(runs):
    rows = [r for _, records in runs for r in records]
    n = len(rows)
    if not n:
        return None
    wrong = [r for r in rows if not r["correct"]]
    return {
        "wrong_answer": sum(r["pred"] is not None for r in wrong) / n,
        "no_answer_truncated": sum(r["pred"] is None and r.get("hit_max_tokens", False) for r in wrong) / n,
        "no_answer_other": sum(r["pred"] is None and not r.get("hit_max_tokens", False) for r in wrong) / n,
        "language_drift": sum(bangla_ratio(r["output"]) < 0.5 for r in rows) / n,
        "repetitive": sum(is_repetitive(r["output"]) for r in rows) / n,
        "repetitive_given_limit": (sum(is_repetitive(r["output"]) for r in rows if r.get("hit_max_tokens"))
                                   / max(1, sum(bool(r.get("hit_max_tokens")) for r in rows))),
        "mean_tokens": statistics.mean(r.get("n_tokens", 0) for r in rows),
    }


def flips(reference, other):
    """Share of reference-correct problems lost, and reference-wrong problems gained."""
    shared = set(reference) & set(other)
    ref_right = [i for i in shared if reference[i] == 1.0]
    ref_wrong = [i for i in shared if reference[i] == 0.0]
    lost = sum(1 - other[i] for i in ref_right) / len(ref_right) if ref_right else 0.0
    gained = sum(other[i] for i in ref_wrong) / len(ref_wrong) if ref_wrong else 0.0
    return lost, gained


def answer_type(gold):
    g = normalize(gold)
    if re.fullmatch(r"[A-E]", g) or re.fullmatch(r"\([A-E]\).+", g):
        return "choice"
    if to_number(g) is not None:
        return "numeric"
    return "expression"


# ----------------------------------------------------------------- aggregation

def aggregate(all_runs):
    reference = per_problem_accuracy(all_runs[REFERENCE]) if REFERENCE in all_runs else None
    out = []
    for method, runs in all_runs.items():
        summaries = [s for s, _ in runs]
        accs = [100 * s["accuracy"] for s in summaries]
        acc, acc_std = mean_std(accs)
        problems = per_problem_accuracy(runs)
        first = summaries[0]
        row = {
            "method": method,
            "label": LABELS.get(method, method),
            "seeds": len(runs),
            "seed_accuracies": accs,
            "n": first["n"],
            "accuracy": acc,
            "accuracy_std": acc_std,
            "boxed_rate": 100 * statistics.mean(s["boxed_rate"] for s in summaries),
            "bits_per_weight": first.get("bits_per_weight"),
            "size_gb": first.get("size_gb", first.get("weights_gb")),
            "device": first.get("device", first.get("gpu", "?")),
            "errors": error_breakdown(runs),
            "problems": problems,
            "summary": first,
        }
        if reference is not None and method != REFERENCE and problems:
            row["delta"] = paired_bootstrap(problems, reference)
            row["lost"], row["gained"] = flips(reference, problems)
        out.append(row)
    holm([r.get("delta") for r in out])  # family: every method vs. the 16-bit reference
    return out


CALIB_PAIRS = {  # macro suffix -> (method, compared against)
    "LangDomain": ("gptq-w4-bangla", "gptq-w4-english"),        # language and domain both differ
    "Lang": ("gptq-w4-bangla", "gptq-w4-english-math"),         # language only
    "Domain": ("gptq-w4-english-math", "gptq-w4-english"),      # domain only
    "GptqRtn": ("gptq-w4-bangla", "rtn-w4"),                    # GPTQ vs plain rounding
    "GptqRtnThree": ("gptq-w3-bangla", "rtn-w3"),
    "LangParallel": ("gptq-w4-bangla-parallel", "gptq-w4-english-parallel"),  # same problems, language only
}


CALIB_KEYS = ("LangDomain", "Lang", "Domain", "LangParallel")


def per_problem_repetition(runs):
    """idx -> fraction of runs (seeds) whose output is a repetition loop."""
    hits = defaultdict(list)
    for _, records in runs:
        for r in records:
            hits[r["idx"]].append(float(is_repetitive(r["output"])))
    return {idx: sum(v) / len(v) for idx, v in hits.items()}


def repetition_comparisons(all_runs):
    """Secondary outcome: does the calibration set change how often outputs loop?"""
    out = {}
    for key in CALIB_KEYS:
        a, b = CALIB_PAIRS[key]
        if a in all_runs and b in all_runs:
            out[key] = (a, b, paired_bootstrap(per_problem_repetition(all_runs[a]), per_problem_repetition(all_runs[b])))
    holm([t for _, _, t in out.values()])
    return out


def comparisons(rows):
    by = {r["method"]: r for r in rows}
    out = {}
    for key, (a, b) in CALIB_PAIRS.items():
        if a in by and b in by:
            out[key] = (a, b, paired_bootstrap(by[a]["problems"], by[b]["problems"]))
    # One Holm family per research question: (1) does GPTQ beat RTN, (2) does the calibration set matter.
    holm([t for k, (_, _, t) in out.items() if k in ("GptqRtn", "GptqRtnThree")])
    holm([t for k, (_, _, t) in out.items() if k not in ("GptqRtn", "GptqRtnThree")])
    return out


# ----------------------------------------------------------------- text output

def fmt_acc(r, tex=False):
    pm = "$\\pm$" if tex else "±"
    return f"{r['accuracy']:.1f}" + (f" {pm} {r['accuracy_std']:.1f}" if r["accuracy_std"] is not None else "")


def pp(x):
    """One decimal, or two when one decimal would round to zero (so -0.02 does not print as -0.0)."""
    if x == 0:
        return "0.0"
    return f"{x:+.2f}" if abs(x) < 0.05 else f"{x:+.1f}"


def fmt_delta(r, tex=False):
    if r.get("delta") is None:
        return "--" if tex else "–"
    d, lo, hi = r["delta"]["d"], r["delta"]["lo"], r["delta"]["hi"]
    mark = "*" if r["delta"]["p_adj"] < 0.05 else ""
    if tex:
        return f"${d:+.1f}${mark} {{\\scriptsize [{pp(lo)}, {pp(hi)}]}}"
    return f"{d:+.1f}{mark} [{pp(lo)}, {pp(hi)}]"


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def fmt_flips(r, tex=False):
    if "lost" not in r:
        return "--" if tex else "–"
    return f"{100 * r['lost']:.0f} / {100 * r['gained']:.0f}"


def markdown(rows, comps):
    lines = [
        f"Held-out problems: {rows[0]['n']} · device: {rows[0]['device']} · "
        f"* = significant after Holm correction (p_adj < 0.05)",
        "",
        "| Method | Seeds | Bits/weight | Size (GB) | Accuracy (%) | Δ vs 16-bit, pp [95% CI] | Lost / gained (%) | Boxed (%) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        bpw = f"{r['bits_per_weight']:.2f}" if r["bits_per_weight"] else "–"
        size = f"{r['size_gb']:.2f}" if r["size_gb"] else "–"
        lines.append(f"| {r['label']} | {r['seeds']} | {bpw} | {size} | {fmt_acc(r)} | {fmt_delta(r)} | "
                     f"{fmt_flips(r)} | {r['boxed_rate']:.0f} |")
    lines += ["", "| Method | Wrong answer (%) | No answer, truncated (%) | No answer, stopped (%) | Language drift (%) | Repetitive (%) | Repetitive among truncated (%) | Mean output tokens |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e = r["errors"]
        if e:
            lines.append(f"| {r['label']} | {100 * e['wrong_answer']:.1f} | {100 * e['no_answer_truncated']:.1f} | "
                         f"{100 * e['no_answer_other']:.1f} | {100 * e['language_drift']:.1f} | {100 * e['repetitive']:.1f} | "
                         f"{100 * e['repetitive_given_limit']:.0f} | {e['mean_tokens']:.0f} |")
    if comps:
        lines += ["", "Paired comparisons:", ""]
        for a, b, t in comps.values():
            lines.append(f"- {LABELS[a]} minus {LABELS[b]}: {t['d']:+.1f} pp [95% CI {pp(t['lo'])}, {pp(t['hi'])}], "
                         f"p = {t['p']:.3f}, Holm p = {t['p_adj']:.3f}")
    return "\n".join(lines)


def tex_main(rows):
    body = []
    for r in rows:
        bpw = f"{r['bits_per_weight']:.2f}" if r["bits_per_weight"] else "--"
        size = f"{r['size_gb']:.2f}" if r["size_gb"] else "--"
        body.append(f"{r['label']} & {bpw} & {size} & {fmt_acc(r, tex=True)} & {fmt_delta(r, tex=True)} & "
                    f"{fmt_flips(r, tex=True)} \\\\")
        if r["method"] in ("16bit", "rtn-w8", "gptq-w4-english-parallel"):
            body.append("\\midrule")
    if body and body[-1] == "\\midrule":
        body.pop()
    return "\n".join(body) + "\n"


def tex_errors(rows):
    body = []
    for r in rows:
        e = r["errors"]
        if e:
            body.append(f"{r['label']} & {100 * e['wrong_answer']:.1f} & {100 * e['no_answer_truncated']:.1f} & "
                        f"{100 * e['no_answer_other']:.1f} & {100 * e['repetitive']:.1f} & {100 * e['language_drift']:.1f} & "
                        f"{e['mean_tokens']:.0f} \\\\")
    return "\n".join(body) + "\n"


def tex_calibration(comps, rows):
    by = {r["method"]: r for r in rows}
    lines = []
    for method, name in [("gptq-w4-bangla", "Bangla"), ("gptq-w4-english-math", "EnglishMath"),
                         ("gptq-w4-english", "EnglishWiki"), ("gptq-w4-bangla-parallel", "BanglaParallel"),
                         ("gptq-w4-english-parallel", "EnglishParallel"), ("rtn-w4", "RtnFour"), ("rtn-w3", "RtnThree"),
                         ("gptq-w3-bangla", "GptqThree"), ("nf4", "NfFour"), ("rtn-w8", "RtnEight")]:
        if method in by:
            lines.append(f"\\newcommand{{\\Acc{name}}}{{{fmt_acc(by[method], tex=True)}}}")
            if by[method].get("delta"):
                lines.append(f"\\newcommand{{\\Drop{name}}}{{${by[method]['delta']['d']:+.1f}$}}")
    for key, (_, _, t) in comps.items():
        lines.append(f"\\newcommand{{\\Delta{key}}}{{${t['d']:+.1f}$~pp (95\\% CI [{pp(t['lo'])}, {pp(t['hi'])}])}}")
        lines.append(f"\\newcommand{{\\P{key}}}{{{fmt_p(t['p_adj'])}}}")
    return "\n".join(lines) + "\n"


def noise_baseline(all_runs):
    """Churn between two 16-bit runs that differ only in batch size (numerical noise, no quantization)."""
    path = RESULTS / "noise" / "16bit-b8.jsonl"
    if REFERENCE not in all_runs or not path.exists():
        return None
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    noise = {"noise": [({"accuracy": 0}, records)]}
    rescore(noise)
    rerun = per_problem_accuracy(noise["noise"])
    reference = per_problem_accuracy(all_runs[REFERENCE])
    lost, gained = flips(reference, rerun)
    acc = 100 * sum(rerun.values()) / len(rerun)
    identical = sum(a["output"] == b["output"] for a, b in zip(
        sorted(records, key=lambda r: r["idx"]), sorted(all_runs[REFERENCE][0][1], key=lambda r: r["idx"])))
    return {"accuracy": acc, "lost": lost, "gained": gained, "identical": identical / len(records)}


def tex_noise(noise):
    if not noise:
        return "% noise baseline not available\n"
    return (f"\\newcommand{{\\NoiseAcc}}{{{noise['accuracy']:.1f}}}\n"
            f"\\newcommand{{\\NoiseLost}}{{{100 * noise['lost']:.0f}}}\n"
            f"\\newcommand{{\\NoiseGained}}{{{100 * noise['gained']:.0f}}}\n"
            f"\\newcommand{{\\NoiseIdentical}}{{{100 * noise['identical']:.0f}}}\n")


def tex_setup(rows):
    by = {r["method"]: r for r in rows}
    first = rows[0]["summary"]
    device = rows[0]["device"]
    device_name = {"mps": "Apple M1 Pro (MPS)", "cuda": "NVIDIA GPU", "cpu": "CPU"}.get(device, device.replace("Tesla ", ""))
    lines = [
        f"\\newcommand{{\\NumEval}}{{{rows[0]['n']}}}",
        f"\\newcommand{{\\NumSeeds}}{{{max(r['seeds'] for r in rows)}}}",
        f"\\newcommand{{\\EvalDevice}}{{{device_name}}}",
        f"\\newcommand{{\\MaxNewTokens}}{{{first.get('max_new_tokens', 512)}}}",
        f"\\newcommand{{\\MaxRefTokens}}{{{first.get('max_ref_tokens') or 'unlimited'}}}",
    ]
    if REFERENCE in by:
        lines.append(f"\\newcommand{{\\RefAcc}}{{{by[REFERENCE]['accuracy']:.1f}}}")
        lines.append(f"\\newcommand{{\\RefSize}}{{{by[REFERENCE]['size_gb']:.2f}}}")
        s = by[REFERENCE]["summary"]
        if s.get("total_params"):
            lines.append(f"\\newcommand{{\\ParamsTotal}}{{{s['total_params'] / 1e9:.2f}}}")
            lines.append(f"\\newcommand{{\\ParamsQuant}}{{{s['quantized_params'] / 1e9:.2f}}}")
            lines.append(f"\\newcommand{{\\ParamsQuantShare}}{{{100 * s['quantized_params'] / s['total_params']:.0f}}}")
    gptq = [r for r in rows if r["method"].startswith("gptq")]
    if gptq and gptq[0]["summary"].get("n_calib"):
        lines.append(f"\\newcommand{{\\NumCalib}}{{{gptq[0]['summary']['n_calib']}}}")
    return "\n".join(lines) + "\n"


def tex_composition(all_runs):
    golds = {}
    for _, records in all_runs.get(REFERENCE, next(iter(all_runs.values()))):
        for rec in records:
            golds[rec["idx"]] = rec["gold"]
    counts = Counter(answer_type(g) for g in golds.values())
    n = max(sum(counts.values()), 1)
    return "".join(
        f"\\newcommand{{\\Comp{k.capitalize()}}}{{{counts.get(k, 0)} ({100 * counts.get(k, 0) / n:.0f}\\%)}}\n"
        for k in ("numeric", "choice", "expression"))


# ----------------------------------------------------------------- figures

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def save(fig, stem):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"{stem}.{ext}", facecolor=SURFACE, dpi=200)
    plt.close(fig)


def plot_tradeoff(rows):
    """One row per method (ordered by size), accuracy as a dot; bar = std over calibration seeds."""
    shown = [r for r in rows if r["size_gb"]]
    if not shown:
        return
    fig, ax = plt.subplots(figsize=(3.4, 0.24 * len(shown) + 0.75))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    y = list(range(len(shown)))[::-1]
    ref = next((r["accuracy"] for r in shown if r["method"] == REFERENCE), None)
    if ref is not None:
        ax.axvline(ref, color=REF_LINE, linewidth=1, zorder=1)
    for yi, r in zip(y, shown):
        if r["accuracy_std"] is not None:
            ax.plot([r["accuracy"] - r["accuracy_std"], r["accuracy"] + r["accuracy_std"]], [yi, yi],
                    color=SERIES[0], linewidth=2, zorder=2)
        ax.scatter(r["accuracy"], yi, s=36, color=SERIES[0], edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.set_yticks(y, [f"{r['label']}  ({r['size_gb']:.2f} GB)" for r in shown], fontsize=6.5, color=INK_SECONDARY)
    ax.set_xlabel("Accuracy (%)", color=INK_SECONDARY, fontsize=8)
    ax.set_xlim(left=0, right=max(r["accuracy"] for r in shown) * 1.15)
    save(fig, "fig_tradeoff")


def plot_calibration(rows, all_runs):
    """Two panels per calibration set: accuracy and share of repetitive outputs; dots = seeds."""
    by = {r["method"]: r for r in rows}
    sets = [("gptq-w4-bangla", "Bangla\nmath"), ("gptq-w4-english-math", "English\nGSM8K"),
            ("gptq-w4-english", "English\nWikipedia"), ("gptq-w4-bangla-parallel", "Bangla\nparallel"),
            ("gptq-w4-english-parallel", "English\nparallel")]
    sets = [(m, l) for m, l in sets if m in by]
    if not sets:
        return
    def seed_repetition(method):
        return [100 * sum(is_repetitive(r["output"]) for r in recs) / len(recs) for _, recs in all_runs[method]]
    panels = [("Accuracy (%)", lambda m: by[m]["seed_accuracies"], "accuracy"),
              ("Repetitive outputs (%)", seed_repetition, "repetition")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    fig.patch.set_facecolor(SURFACE)
    for ax, (ylabel, values, kind) in zip(axes, panels):
        style(ax)
        for x, (m, _) in enumerate(sets):
            vals = values(m)
            ax.plot([x - 0.25, x + 0.25], [statistics.mean(vals)] * 2, color=SERIES[0], linewidth=2, zorder=3)
            jitter = [x + (i - (len(vals) - 1) / 2) * 0.1 for i in range(len(vals))]
            ax.scatter(jitter, vals, s=24, color=SERIES[0], edgecolors=SURFACE, linewidths=1.5, zorder=4)
        if REFERENCE in by:
            ref = by[REFERENCE]["accuracy"] if kind == "accuracy" else \
                100 * by[REFERENCE]["errors"]["repetitive"]
            ax.axhline(ref, color=REF_LINE, linewidth=1, zorder=1)
            ax.annotate("16-bit", (len(sets) - 0.5, ref), textcoords="offset points", xytext=(0, 3),
                        fontsize=6.5, color=INK_SECONDARY, ha="right")
        ax.axvline(2.5, color=GRID, linewidth=1, zorder=0)  # separates the parallel pair
        ax.set_xticks(range(len(sets)), [l for _, l in sets], fontsize=6.5, color=INK_SECONDARY)
        ax.set_xlim(-0.5, len(sets) - 0.5)
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8)
    save(fig, "fig_calibration")


def plot_errors(rows):
    shown = [r for r in rows if r["errors"]]
    if not shown:
        return
    parts = [("wrong_answer", "Wrong answer"), ("no_answer_truncated", "Token limit"),
             ("no_answer_other", "Stopped early")]
    fig, ax = plt.subplots(figsize=(3.4, 0.25 * len(shown) + 1.0))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    y = list(range(len(shown)))[::-1]
    left = [0.0] * len(shown)
    for (key, label), color in zip(parts, SERIES):
        vals = [100 * r["errors"][key] for r in shown]
        ax.barh(y, vals, left=left, height=0.7, color=color, edgecolor=SURFACE, linewidth=1, label=label)
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(y, [r["label"] for r in shown], fontsize=6.5, color=INK_SECONDARY)
    ax.set_xlabel("Share of problems (%)", color=INK_SECONDARY, fontsize=8)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=3,
              labelcolor=INK_SECONDARY, handlelength=1, columnspacing=1)
    save(fig, "fig_errors")


# ----------------------------------------------------------------- main

def main():
    all_runs = load()
    if not all_runs:
        raise SystemExit("No results/*.json found. Run src/simulate.py or src/evaluate.py first.")
    rows = aggregate(all_runs)
    comps = comparisons(rows)

    md = markdown(rows, comps)
    (RESULTS / "summary.md").write_text(md + "\n", encoding="utf-8")
    plot_tradeoff(rows)
    plot_calibration(rows, all_runs)
    plot_errors(rows)

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "main.tex").write_text(tex_main(rows), encoding="utf-8")
    (TABLES / "errors.tex").write_text(tex_errors(rows), encoding="utf-8")
    (TABLES / "calibration.tex").write_text(tex_calibration(comps, rows), encoding="utf-8")
    (TABLES / "setup.tex").write_text(tex_setup(rows), encoding="utf-8")
    (TABLES / "composition.tex").write_text(tex_composition(all_runs), encoding="utf-8")
    reps = repetition_comparisons(all_runs)
    rep_lines = []
    for key, (a, b, t) in reps.items():
        rep_lines.append(f"\\newcommand{{\\RepDelta{key}}}{{${t['d']:+.1f}$~pp (95\\% CI [{pp(t['lo'])}, {pp(t['hi'])}])}}")
        rep_lines.append(f"\\newcommand{{\\RepP{key}}}{{{fmt_p(t['p_adj'])}}}")
        md += f"\n- Repetition, {LABELS[a]} minus {LABELS[b]}: {t['d']:+.1f} pp [{pp(t['lo'])}, {pp(t['hi'])}], Holm p = {t['p_adj']:.3f}"
    (TABLES / "repetition.tex").write_text("\n".join(rep_lines) + "\n", encoding="utf-8")
    (RESULTS / "summary.md").write_text(md + "\n", encoding="utf-8")
    noise = noise_baseline(all_runs)
    (TABLES / "noise.tex").write_text(tex_noise(noise), encoding="utf-8")
    if noise:
        md += (f"\n\nNoise baseline (16-bit, batch 8 vs 16): accuracy {noise['accuracy']:.1f}%, "
               f"lost {100 * noise['lost']:.0f}% / gained {100 * noise['gained']:.0f}%, "
               f"identical outputs {100 * noise['identical']:.0f}%")
        (RESULTS / "summary.md").write_text(md + "\n", encoding="utf-8")

    print(md)
    print(f"\nWrote {RESULTS}/summary.md, {RESULTS}/fig_*.pdf and {TABLES}/*.tex")


if __name__ == "__main__":
    main()
