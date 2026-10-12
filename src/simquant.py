"""Simulated weight quantization: quantize, then store the dequantized values in 16-bit.

The model computes with exactly the rounded weights a real low-bit kernel would use, so
accuracy is the same as with packed weights, but memory and speed are not reduced. This is
the standard way to measure accuracy in quantization research and it runs on any device
(CPU, Apple MPS, CUDA). Sizes are computed from the bit widths instead of being measured.

Methods
  rtn   round-to-nearest, asymmetric, per group (or per output channel if group_size = -1)
  nf4   bitsandbytes NF4: 16 normal-distribution levels, absmax per block of 64 weights
  gptq  GPTQ (Frantar et al., 2023), following the reference implementation
        (github.com/IST-DASLab/gptq): layer by layer, each layer calibrated on the outputs
        of the already-quantized layers before it.

Embeddings, norms and the (tied) output head stay in 16-bit, as in GPTQ and bitsandbytes.
"""

import math

import torch
import torch.nn as nn

# bitsandbytes NF4 code book (QLoRA, Appendix E)
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
])
NF4_BLOCK = 64


def decoder_linears(model):
    """name -> nn.Linear for every projection inside the transformer blocks."""
    return {name: m for name, m in model.model.layers.named_modules() if isinstance(m, nn.Linear)}


# ----------------------------------------------------------------- round to nearest

def asym_params(w, bits):
    """Scale and zero point per row of w, for asymmetric quantization to 2**bits levels."""
    maxq = 2 ** bits - 1
    wmin = w.min(dim=1, keepdim=True).values.clamp(max=0)
    wmax = w.max(dim=1, keepdim=True).values.clamp(min=0)
    same = (wmax - wmin) == 0
    wmin, wmax = torch.where(same, -torch.ones_like(wmin), wmin), torch.where(same, torch.ones_like(wmax), wmax)
    scale = (wmax - wmin) / maxq
    zero = torch.round(-wmin / scale)
    return scale, zero


def fake_quant(w, scale, zero, bits):
    q = torch.clamp(torch.round(w / scale) + zero, 0, 2 ** bits - 1)
    return scale * (q - zero)


@torch.no_grad()
def rtn_weight(w, bits, group_size):
    w = w.float()
    rows, cols = w.shape
    if group_size <= 0:
        return fake_quant(w, *asym_params(w, bits), bits)
    g = w.reshape(rows, cols // group_size, group_size).reshape(-1, group_size)
    out = fake_quant(g, *asym_params(g, bits), bits)
    return out.reshape(rows, cols)


@torch.no_grad()
def nf4_weight(w):
    flat = w.float().flatten()
    pad = (-flat.numel()) % NF4_BLOCK
    blocks = torch.cat([flat, flat.new_zeros(pad)]).reshape(-1, NF4_BLOCK)
    absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    levels = NF4_LEVELS.to(w.device)
    idx = (blocks / absmax).unsqueeze(-1).sub(levels).abs().argmin(dim=-1)
    deq = levels[idx] * absmax
    return deq.flatten()[:flat.numel()].reshape(w.shape)


@torch.no_grad()
def apply_rtn(model, bits, group_size):
    for layer in decoder_linears(model).values():
        layer.weight.data = rtn_weight(layer.weight.data, bits, group_size).to(layer.weight.dtype)


@torch.no_grad()
def apply_nf4(model):
    for layer in decoder_linears(model).values():
        layer.weight.data = nf4_weight(layer.weight.data).to(layer.weight.dtype)


# ----------------------------------------------------------------- GPTQ

class GPTQ:
    """Hessian accumulation and the GPTQ update for one linear layer."""

    def __init__(self, layer):
        self.layer = layer
        self.cols = layer.weight.shape[1]
        self.H = torch.zeros(self.cols, self.cols, device=layer.weight.device, dtype=torch.float32)
        self.n = 0

    def add_batch(self, inp):
        x = inp.reshape(-1, self.cols).float()
        rows = x.shape[0]
        self.H *= self.n / (self.n + rows)
        self.n += rows
        x = math.sqrt(2 / self.n) * x
        self.H += x.t() @ x

    @torch.no_grad()
    def quantize(self, bits, group_size=128, blocksize=128, percdamp=0.01):
        # The column loop runs many small ops, which is faster on CPU than on MPS.
        W = self.layer.weight.data.float().cpu().clone()
        H = self.H.cpu()
        self.H = None

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        H += damp * torch.eye(self.cols)
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        Q = torch.zeros_like(W)
        scale = zero = None
        if group_size <= 0:
            scale, zero = asym_params(W, bits)

        for i1 in range(0, self.cols, blocksize):
            i2 = min(i1 + blocksize, self.cols)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(i2 - i1):
                col = i1 + i
                if group_size > 0 and col % group_size == 0:
                    # W1 holds the error-corrected values for the current block.
                    group = torch.cat([W1[:, i:], W[:, i2:]], dim=1)[:, :group_size]
                    scale, zero = asym_params(group, bits)
                w = W1[:, i]
                d = Hinv1[i, i]
                q = fake_quant(w.unsqueeze(1), scale, zero, bits).flatten()
                Q1[:, i] = q
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        self.layer.weight.data = Q.to(self.layer.weight.device, self.layer.weight.dtype)


class _Catcher(nn.Module):
    """Stands in for the first decoder layer to record its inputs, then stops the forward pass."""

    def __init__(self, layer, store):
        super().__init__()
        self.layer = layer
        self.store = store

    def forward(self, hidden_states, *args, **kwargs):
        self.store.append((hidden_states, args, kwargs))
        raise ValueError("captured")


@torch.no_grad()
def apply_gptq(model, tokenizer, texts, bits, group_size=128, max_tokens=512, log=print):
    device = next(model.parameters()).device
    layers = model.model.layers

    # 1. Inputs to the first decoder layer, one entry per calibration sample.
    inputs = []
    layers[0] = _Catcher(layers[0], inputs)
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(device)
        try:
            model(ids, use_cache=False)
        except ValueError:
            pass
    layers[0] = layers[0].layer

    # 2. Quantize layer by layer; each layer sees the outputs of the quantized layers before it.
    for li, layer in enumerate(layers):
        linears = {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}
        gptq = {name: GPTQ(m) for name, m in linears.items()}
        hooks = [m.register_forward_hook(lambda mod, inp, out, name=name: gptq[name].add_batch(inp[0]))
                 for name, m in linears.items()]
        for hidden, args, kwargs in inputs:
            layer(hidden, *args, **kwargs)
        for h in hooks:
            h.remove()

        for name in linears:
            gptq[name].quantize(bits, group_size)
        del gptq

        inputs = [(layer(hidden, *args, **kwargs)[0], args, kwargs) for hidden, args, kwargs in inputs]
        log(f"  GPTQ layer {li + 1}/{len(layers)} done")

    if device.type == "mps":
        torch.mps.empty_cache()


# ----------------------------------------------------------------- size accounting

def bits_per_weight(method, bits, group_size):
    """Average storage bits per quantized weight, including scales and zero points."""
    if method == "16bit":
        return 16.0
    if method == "nf4":
        return 4 + 32 / NF4_BLOCK            # fp32 absmax per block (no double quantization)
    if group_size <= 0:
        return bits + 32 / 2304             # fp16 scale + zero per output row; negligible
    return bits + (16 + bits) / group_size  # fp16 scale + packed zero point per group


def model_size_gb(model, method, bits, group_size):
    quantized = sum(m.weight.numel() for m in decoder_linears(model).values())
    total = sum(p.numel() for p in model.parameters())
    bits_total = quantized * bits_per_weight(method, bits, group_size) + (total - quantized) * 16
    return bits_total / 8 / 1024**3, quantized, total
