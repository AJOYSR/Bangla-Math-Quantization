Held-out problems: 200 · device: mps · * = significant after Holm correction (p_adj < 0.05)

| Method | Seeds | Bits/weight | Size (GB) | Accuracy (%) | Δ vs 16-bit, pp [95% CI] | Lost / gained (%) | Boxed (%) |
|---|---|---|---|---|---|---|---|
| 16-bit (reference) | 1 | 16.00 | 4.87 | 17.5 | – | – | 57 |
| RTN 8-bit | 1 | 8.01 | 2.99 | 17.5 | +0.0 [-4.0, +4.0] | 23 / 5 | 56 |
| NF4 4-bit | 1 | 4.50 | 2.16 | 15.0 | -2.5 [-8.0, +3.0] | 51 / 8 | 60 |
| RTN 4-bit | 1 | 4.16 | 2.08 | 10.5 | -7.0 [-12.5, -1.5] | 66 / 5 | 44 |
| GPTQ 4-bit, Bangla math | 3 | 4.16 | 2.08 | 16.7 ± 1.2 | -0.8 [-5.2, +3.3] | 45 / 8 | 57 |
| GPTQ 4-bit, English GSM8K | 3 | 4.16 | 2.08 | 13.8 ± 2.3 | -3.7 [-8.5, +1.0] | 59 / 8 | 50 |
| GPTQ 4-bit, English wiki | 3 | 4.16 | 2.08 | 13.0 ± 0.9 | -4.5 [-9.2, 0.0] | 58 / 7 | 50 |
| GPTQ 4-bit, Bangla parallel | 3 | 4.16 | 2.08 | 17.3 ± 0.8 | -0.2 [-4.3, +4.0] | 42 / 9 | 57 |
| GPTQ 4-bit, English parallel | 3 | 4.16 | 2.08 | 15.7 ± 3.3 | -1.8 [-6.7, +3.0] | 57 / 10 | 49 |
| RTN 3-bit | 1 | 3.15 | 1.84 | 0.0 | -17.5* [-23.0, -12.5] | 100 / 0 | 0 |
| GPTQ 3-bit, Bangla math | 1 | 3.15 | 1.84 | 14.5 | -3.0 [-8.5, +2.5] | 54 / 8 | 48 |

| Method | Wrong answer (%) | No answer, truncated (%) | No answer, stopped (%) | Language drift (%) | Repetitive (%) | Repetitive among truncated (%) | Mean output tokens |
|---|---|---|---|---|---|---|---|
| 16-bit (reference) | 39.5 | 39.5 | 3.5 | 0.5 | 44.5 | 87 | 499 |
| RTN 8-bit | 38.5 | 41.5 | 2.5 | 0.5 | 42.0 | 82 | 499 |
| NF4 4-bit | 44.5 | 37.0 | 3.5 | 0.5 | 44.5 | 93 | 505 |
| RTN 4-bit | 33.5 | 52.5 | 3.5 | 0.5 | 58.5 | 95 | 565 |
| GPTQ 4-bit, Bangla math | 40.8 | 39.0 | 3.5 | 0.0 | 44.2 | 88 | 510 |
| GPTQ 4-bit, English GSM8K | 35.7 | 46.3 | 4.2 | 0.0 | 53.8 | 93 | 537 |
| GPTQ 4-bit, English wiki | 36.5 | 46.2 | 4.3 | 0.0 | 52.3 | 91 | 536 |
| GPTQ 4-bit, Bangla parallel | 40.0 | 39.2 | 3.5 | 0.0 | 43.0 | 85 | 520 |
| GPTQ 4-bit, English parallel | 33.3 | 47.2 | 3.8 | 0.0 | 53.2 | 95 | 538 |
| RTN 3-bit | 0.5 | 98.5 | 1.0 | 0.0 | 97.5 | 99 | 761 |
| GPTQ 3-bit, Bangla math | 34.0 | 46.0 | 5.5 | 0.0 | 53.5 | 97 | 536 |

Paired comparisons:

- GPTQ 4-bit, Bangla math minus GPTQ 4-bit, English wiki: +3.7 pp [95% CI +0.7, +6.8], p = 0.020, Holm p = 0.078
- GPTQ 4-bit, Bangla math minus GPTQ 4-bit, English GSM8K: +2.8 pp [95% CI -0.5, +6.2], p = 0.100, Holm p = 0.301
- GPTQ 4-bit, English GSM8K minus GPTQ 4-bit, English wiki: +0.8 pp [95% CI -2.5, +4.0], p = 0.656, Holm p = 0.782
- GPTQ 4-bit, Bangla math minus RTN 4-bit: +6.2 pp [95% CI +1.7, +10.7], p = 0.007, Holm p = 0.007
- GPTQ 3-bit, Bangla math minus RTN 3-bit: +14.5 pp [95% CI +10.0, +19.5], p = 0.000, Holm p = 0.000
- GPTQ 4-bit, Bangla parallel minus GPTQ 4-bit, English parallel: +1.7 pp [95% CI -2.0, +5.3], p = 0.391, Holm p = 0.782
- Repetition, GPTQ 4-bit, Bangla math minus GPTQ 4-bit, English wiki: -8.2 pp [-13.3, -3.2], Holm p = 0.002
- Repetition, GPTQ 4-bit, Bangla math minus GPTQ 4-bit, English GSM8K: -9.7 pp [-15.0, -4.5], Holm p = 0.002
- Repetition, GPTQ 4-bit, English GSM8K minus GPTQ 4-bit, English wiki: +1.5 pp [-3.7, +6.7], Holm p = 0.575
- Repetition, GPTQ 4-bit, Bangla parallel minus GPTQ 4-bit, English parallel: -10.2 pp [-15.2, -5.2], Holm p = 0.000

Noise baseline (16-bit, batch 8 vs 16): accuracy 17.0%, lost 3% / gained 0%, identical outputs 87%
