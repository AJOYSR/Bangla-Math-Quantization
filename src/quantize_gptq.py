"""GPTQ post-training quantization of the merged model.

GPTQ (Frantar, Ashkboos, Hoefler, Alistarh; ICLR 2023) quantizes one layer at a time
and uses a small calibration set to correct the rounding error of each weight column.
The calibration set is the variable here. Comparing Bangla math, English math and
generic English text separates the effect of calibration language from that of domain.

    python src/quantize_gptq.py --bits 4 --calib bangla  --seed 0 --out models/gptq-w4-bangla-s0
    python src/quantize_gptq.py --bits 4 --calib english-math --seed 0 --out models/gptq-w4-english-math-s0
    python src/quantize_gptq.py --bits 4 --calib english --seed 0 --out models/gptq-w4-english-s0
    python src/quantize_gptq.py --bits 3 --calib bangla  --seed 0 --out models/gptq-w3-bangla-s0
"""

import argparse
import time

from gptqmodel import GPTQModel, QuantizeConfig

from common import load_calibration_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/merged-16bit")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calib", default="bangla", choices=["bangla", "english-math", "english", "bangla-parallel", "english-parallel"])
    parser.add_argument("--n-calib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0, help="which calibration samples are drawn")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    calibration = load_calibration_texts(args.calib, n=args.n_calib, seed=args.seed)
    print(f"{len(calibration)} {args.calib} calibration samples (seed {args.seed})")

    config = QuantizeConfig(bits=args.bits, group_size=args.group_size)
    model = GPTQModel.load(args.model, config)

    start = time.time()
    model.quantize(calibration, batch_size=1)
    print(f"Quantized in {(time.time() - start) / 60:.1f} min")

    model.save(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
