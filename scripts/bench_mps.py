"""M0 benchmark: training-step throughput and memory for a *stand-in* GPT on this machine.

The stand-in uses torch.nn.TransformerEncoderLayer + a causal mask, with the same shapes as the
planned presets, so FLOPs/memory match the model you will build in A3/A4 to first order.
It exists only to validate the planning estimates in docs/STATE.md — it is not your model.

Examples:
    python scripts/bench_mps.py                       # base preset, ~60 timed steps
    python scripts/bench_mps.py --preset stretch
    python scripts/bench_mps.py --duration 600        # 10-min sustained run: watch thermal throttling
    python scripts/bench_mps.py --device cpu --preset debug
"""

from __future__ import annotations

import argparse
import platform
import resource
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    "debug": dict(vocab=4096, ctx=128, d_model=128, n_layer=2, n_head=4, batch=32),
    "base": dict(vocab=4096, ctx=256, d_model=384, n_layer=6, n_head=6, batch=32),
    "stretch": dict(vocab=4096, ctx=256, d_model=512, n_layer=8, n_head=8, batch=32),
}


class StandInGPT(nn.Module):
    def __init__(self, vocab: int, ctx: int, d_model: int, n_layer: int, n_head: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, 4 * d_model, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        nn.init.normal_(self.tok.weight, std=0.02)  # N(0,1) default would give huge initial logits
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def device_mem_bytes(device: torch.device) -> int | None:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    return None


def process_peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024  # macOS: bytes, Linux: KiB


def matmul_tflops(device: torch.device, dtype: torch.dtype, n: int = 2048, iters: int = 20) -> float | None:
    try:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        for _ in range(3):
            a @ b
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            a @ b
        sync(device)
        return 2 * n**3 * iters / (time.perf_counter() - t0) / 1e12
    except (RuntimeError, TypeError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default="base")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, help="override micro-batch size")
    p.add_argument("--steps", type=int, default=60, help="timed steps (ignored if --duration)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--duration", type=float, help="run for this many seconds, reporting per window")
    p.add_argument("--window", type=float, default=30.0, help="report interval (s) with --duration")
    p.add_argument("--skip-matmul", action="store_true")
    args = p.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.batch:
        cfg["batch"] = args.batch
    device = pick_device(args.device)
    torch.manual_seed(0)

    print(f"torch {torch.__version__} | python {platform.python_version()} | {platform.platform()}")
    print(f"device: {device} | preset: {args.preset} | {cfg}")

    if not args.skip_matmul:
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            tf = matmul_tflops(device, dt)
            print(f"matmul 2048² {str(dt):15s}: " + (f"{tf:6.2f} TFLOP/s" if tf else "unsupported"))

    batch, ctx = cfg.pop("batch"), cfg["ctx"]
    model = StandInGPT(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())  # tied weight counted once
    n_pos = model.pos.weight.numel()
    flops_per_token = 6 * (n_params - n_pos) + 12 * cfg["n_layer"] * ctx * cfg["d_model"]
    print(f"params: {n_params/1e6:.2f} M | est. train FLOPs/token: {flops_per_token/1e6:.1f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tokens_per_step = batch * ctx

    def step() -> float:
        x = torch.randint(0, cfg["vocab"], (batch, ctx), device=device)
        y = torch.randint(0, cfg["vocab"], (batch, ctx), device=device)
        loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    with torch.no_grad():
        x0 = torch.randint(0, cfg["vocab"], (batch, ctx), device=device)
        loss0 = model(x0, torch.randint(0, cfg["vocab"], (batch, ctx), device=device)).item()
    print(f"initial loss ≈ {loss0:.3f}  (ln vocab = {torch.log(torch.tensor(float(cfg['vocab']))).item():.3f})")
    for _ in range(args.warmup):
        step()
    sync(device)

    peak_dev = 0
    start = time.perf_counter()
    win_start, win_steps, total_steps = start, 0, 0
    rates = []
    while True:
        step()
        total_steps += 1
        win_steps += 1
        if args.duration is None and total_steps >= args.steps:
            break
        now = time.perf_counter()
        if args.duration is not None and now - win_start >= args.window:
            sync(device)
            now = time.perf_counter()
            rate = win_steps * tokens_per_step / (now - win_start)
            rates.append(rate)
            m = device_mem_bytes(device)
            peak_dev = max(peak_dev, m or 0)
            print(f"  t={now - start:6.0f}s  {rate:9.0f} tok/s")
            win_start, win_steps = now, 0
            if now - start >= args.duration:
                break
    sync(device)
    elapsed = time.perf_counter() - start
    m = device_mem_bytes(device)
    peak_dev = max(peak_dev, m or 0)

    tps = total_steps * tokens_per_step / elapsed
    print(f"\nthroughput: {tps:,.0f} tok/s  ({elapsed / total_steps * 1e3:.1f} ms/step, {total_steps} steps)")
    print(f"achieved: {tps * flops_per_token / 1e12:.2f} TFLOP/s | tokens/hour: {tps * 3600 / 1e6:.1f} M")
    if len(rates) >= 2:
        print(f"sustained: first window {rates[0]:,.0f} → last window {rates[-1]:,.0f} tok/s "
              f"({(rates[-1] / rates[0] - 1) * 100:+.1f}%)")
    if peak_dev:
        print(f"device memory (driver/allocator): {peak_dev / 2**30:.2f} GiB")
    print(f"process peak RSS: {process_peak_rss_bytes() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
