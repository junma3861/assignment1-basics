"""
Learning Rate Hyperparameter Sweep for TinyStoriesV2.

Fixed architecture:
  context_length : 256
  d_model        : 512
  d_ff           : 1344
  rope_theta     : 10000
  num_layers     : 4
  num_heads      : 16
  vocab_size     : 10000

Sweeps over a log-spaced grid of learning rates and reports the final
train / validation losses (or flags divergence when the loss goes NaN/Inf).
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── project imports ──────────────────────────────────────────────────────────
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adam_w import AdamW
from cs336_basics.data_loading import get_batch
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import clip_gradients

# ═══════════════════════════════════════════════════════════════════════════
#  FIXED HYPER-PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent          # assignment1-basics/

DATA_TRAIN = BASE_DIR / "data" / "TinyStoriesV2-GPT4-train.uint16"
DATA_VAL   = BASE_DIR / "data" / "TinyStoriesV2-GPT4-valid.uint16"

BPE_VOCAB_SIZE  = 10_000

# Model
CONTEXT_LENGTH  = 256
D_MODEL         = 512
D_FF            = 1_344
ROPE_THETA      = 10_000.0
NUM_LAYERS      = 4
NUM_HEADS       = 16

# Training loop
BATCH_SIZE      = 64
MAX_ITERS       = 5_000        # iterations per LR run (tunable)
WARMUP_ITERS    = 500
MIN_LR_RATIO    = 0.1          # min_lr = lr * MIN_LR_RATIO
WEIGHT_DECAY    = 0.1
BETA1, BETA2    = 0.9, 0.95
EPS             = 1e-8
GRAD_CLIP       = 1.0
LOG_INTERVAL    = 100
EVAL_INTERVAL   = 1_000
EVAL_ITERS      = 50           # batches used for val loss estimate
SEED            = 42

# Learning rates to sweep
LEARNING_RATES  = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_memmap(path: Path) -> np.ndarray:
    return np.memmap(str(path), dtype=np.uint16, mode="r")


@torch.no_grad()
def estimate_loss(model, train_data, val_data) -> dict[str, float]:
    model.eval()
    results = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            losses.append(loss.item())
        results[name] = float(np.mean(losses))
    model.train()
    return results


def is_diverged(loss: float) -> bool:
    return math.isnan(loss) or math.isinf(loss) or loss > 20.0


# ═══════════════════════════════════════════════════════════════════════════
#  SINGLE-RUN TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def run_one_lr(lr: float, train_data: np.ndarray, val_data: np.ndarray) -> dict:
    """Train for MAX_ITERS steps with a given LR and return result dict."""
    set_seed(SEED)

    model = TransformerLM(
        vocab_size=BPE_VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        rope_theta=ROPE_THETA,
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(BETA1, BETA2),
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )

    min_lr = lr * MIN_LR_RATIO
    diverged = False
    recent_losses: list[float] = []
    val_losses_history: list[tuple[int, float]] = []

    print(f"\n{'='*70}")
    print(f"  LR = {lr:.2e}   |  device = {DEVICE}  |  iters = {MAX_ITERS}")
    print(f"{'='*70}")

    model.train()
    t0 = time.time()

    for step in range(MAX_ITERS):
        # ── cosine LR schedule with warmup ──────────────────────────────
        step_lr = get_lr_cosine_schedule(
            step,
            max_learning_rate=lr,
            min_learning_rate=min_lr,
            warmup_iters=WARMUP_ITERS,
            cosine_cycle_iters=MAX_ITERS,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = step_lr

        # ── forward / backward ───────────────────────────────────────────
        x, y = get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model.parameters(), GRAD_CLIP)
        optimizer.step()

        loss_val = loss.item()
        recent_losses.append(loss_val)

        # ── divergence check ─────────────────────────────────────────────
        if is_diverged(loss_val):
            diverged = True
            print(f"  [step {step+1:5d}] DIVERGED  (loss={loss_val})")
            break

        # ── logging ──────────────────────────────────────────────────────
        if (step + 1) % LOG_INTERVAL == 0:
            avg = float(np.mean(recent_losses[-LOG_INTERVAL:]))
            elapsed = time.time() - t0
            tps = BATCH_SIZE * CONTEXT_LENGTH * LOG_INTERVAL / elapsed
            t0 = time.time()
            print(
                f"  step {step+1:5d}/{MAX_ITERS} | "
                f"loss {avg:.4f} | "
                f"lr {step_lr:.2e} | "
                f"{tps:,.0f} tok/s"
            )

        # ── periodic validation ───────────────────────────────────────────
        if (step + 1) % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            val_losses_history.append((step + 1, losses["val"]))
            print(
                f"  [eval @ {step+1}]  train={losses['train']:.4f}  "
                f"val={losses['val']:.4f}"
            )

    # ── final evaluation ─────────────────────────────────────────────────
    if not diverged:
        final = estimate_loss(model, train_data, val_data)
        print(
            f"\n  FINAL  train={final['train']:.4f}  val={final['val']:.4f}"
        )
    else:
        final = {"train": float("nan"), "val": float("nan")}

    # free GPU memory before next run
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "lr": lr,
        "diverged": diverged,
        "final_train_loss": final["train"],
        "final_val_loss": final["val"],
        "val_history": val_losses_history,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SWEEP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  TinyStoriesV2 – Learning Rate Sweep")
    print(f"  device      : {DEVICE}")
    print(f"  model       : {NUM_LAYERS}L {D_MODEL}d {NUM_HEADS}h  d_ff={D_FF}")
    print(f"  context     : {CONTEXT_LENGTH}   vocab: {BPE_VOCAB_SIZE}")
    print(f"  iterations  : {MAX_ITERS}  batch: {BATCH_SIZE}")
    print(f"  LRs to test : {LEARNING_RATES}")
    print("=" * 70)

    # Load data once (memory-mapped, so basically free)
    print("\nLoading data …")
    train_data = load_memmap(DATA_TRAIN)
    val_data   = load_memmap(DATA_VAL)
    print(f"  train tokens : {len(train_data):,}")
    print(f"  val   tokens : {len(val_data):,}")

    results = []
    for lr in LEARNING_RATES:
        result = run_one_lr(lr, train_data, val_data)
        results.append(result)

    # ── Summary table ──────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  LEARNING RATE SWEEP SUMMARY")
    print("=" * 70)
    print(f"  {'LR':>10}  {'Train Loss':>12}  {'Val Loss':>12}  {'Status':>12}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")
    for r in results:
        status = "DIVERGED" if r["diverged"] else "OK"
        tl = f"{r['final_train_loss']:.4f}" if not r["diverged"] else "  N/A"
        vl = f"{r['final_val_loss']:.4f}" if not r["diverged"] else "  N/A"
        print(f"  {r['lr']:>10.2e}  {tl:>12}  {vl:>12}  {status:>12}")
    print("=" * 70)

    # Best (non-diverged) run
    valid = [r for r in results if not r["diverged"]]
    if valid:
        best = min(valid, key=lambda r: r["final_val_loss"])
        print(
            f"\n  Best LR: {best['lr']:.2e}  "
            f"(val loss = {best['final_val_loss']:.4f})"
        )


if __name__ == "__main__":
    main()
