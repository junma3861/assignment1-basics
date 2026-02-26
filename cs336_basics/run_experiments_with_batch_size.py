"""
Batch Size Hyperparameter Sweep for TinyStoriesV2.

Fixed architecture (same as run_experiments_with_learning_rate.py):
  context_length : 256
  d_model        : 512
  d_ff           : 1344
  rope_theta     : 10000
  num_layers     : 4
  num_heads      : 16
  vocab_size     : 10000

Fixed learning rate: lr = 3.00e-03  (best from the LR sweep)

Sweeps over batch sizes from 1 up to the GPU memory limit and reports
final train / validation losses (flags OOM or divergence as needed).
"""

import math
import sys
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

# Model (identical to LR sweep)
CONTEXT_LENGTH  = 256
D_MODEL         = 512
D_FF            = 1_344
ROPE_THETA      = 10_000.0
NUM_LAYERS      = 4
NUM_HEADS       = 16

# Optimal LR from the LR sweep
LEARNING_RATE   = 3e-3

# Training loop (shared settings)
MAX_ITERS       = 5_000        # iterations per batch-size run
WARMUP_ITERS    = 500
MIN_LR_RATIO    = 0.1          # min_lr = LEARNING_RATE * MIN_LR_RATIO
WEIGHT_DECAY    = 0.1
BETA1, BETA2    = 0.9, 0.95
EPS             = 1e-8
GRAD_CLIP       = 1.0
LOG_INTERVAL    = 100
EVAL_INTERVAL   = 1_000
EVAL_ITERS      = 50           # batches used for val loss estimate
SEED            = 42

# Batch sizes to sweep: 1, 8, 16, 32, 64, 128, 256, 512
# (script will skip sizes that exceed GPU memory)
BATCH_SIZES = [1, 8, 16, 32, 64, 128]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOG_DIR  = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "batch_size_sweep.log"

# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING HELPER
# ═══════════════════════════════════════════════════════════════════════════

class Tee:
    """Mirror every print() call to both stdout and a log file."""
    def __init__(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(filepath, "w", buffering=1, encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

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
def estimate_loss(model, train_data, val_data, batch_size: int) -> dict[str, float]:
    model.eval()
    results = {}
    eval_bs = min(batch_size, 64)   # cap eval batch size to avoid OOM
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(data, eval_bs, CONTEXT_LENGTH, DEVICE)
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

def run_one_batch_size(
    batch_size: int,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> dict:
    """Train for MAX_ITERS steps with a given batch size and return result dict."""
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
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )

    min_lr = LEARNING_RATE * MIN_LR_RATIO
    diverged = False
    oom = False
    recent_losses: list[float] = []
    val_losses_history: list[tuple[int, float]] = []

    print(f"\n{'='*70}")
    print(
        f"  Batch Size = {batch_size:4d}  |  lr = {LEARNING_RATE:.2e}  |  "
        f"device = {DEVICE}  |  iters = {MAX_ITERS}"
    )
    print(f"{'='*70}")

    model.train()
    t0 = time.time()

    for step in range(MAX_ITERS):
        # ── cosine LR schedule with warmup ──────────────────────────────
        step_lr = get_lr_cosine_schedule(
            step,
            max_learning_rate=LEARNING_RATE,
            min_learning_rate=min_lr,
            warmup_iters=WARMUP_ITERS,
            cosine_cycle_iters=MAX_ITERS,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = step_lr

        # ── forward / backward ───────────────────────────────────────────
        try:
            x, y = get_batch(train_data, batch_size, CONTEXT_LENGTH, DEVICE)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            clip_gradients(model.parameters(), GRAD_CLIP)
            optimizer.step()
        except torch.cuda.OutOfMemoryError:
            oom = True
            print(f"  [step {step+1:5d}] OUT OF MEMORY — skipping this batch size.")
            break

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
            tps = batch_size * CONTEXT_LENGTH * LOG_INTERVAL / elapsed
            t0 = time.time()
            print(
                f"  step {step+1:5d}/{MAX_ITERS} | "
                f"loss {avg:.4f} | "
                f"lr {step_lr:.2e} | "
                f"{tps:,.0f} tok/s"
            )

        # ── periodic validation ───────────────────────────────────────────
        if (step + 1) % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data, batch_size)
            val_losses_history.append((step + 1, losses["val"]))
            print(
                f"  [eval @ {step+1}]  train={losses['train']:.4f}  "
                f"val={losses['val']:.4f}"
            )

    # ── final evaluation ─────────────────────────────────────────────────
    if not diverged and not oom:
        final = estimate_loss(model, train_data, val_data, batch_size)
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
        "batch_size": batch_size,
        "diverged": diverged,
        "oom": oom,
        "final_train_loss": final["train"],
        "final_val_loss": final["val"],
        "val_history": val_losses_history,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SWEEP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    tee = Tee(LOG_FILE)
    sys.stdout = tee
    print(f"Logging to: {LOG_FILE}")

    print("\n" + "=" * 70)
    print("  TinyStoriesV2 – Batch Size Sweep")
    print(f"  device        : {DEVICE}")
    print(f"  model         : {NUM_LAYERS}L {D_MODEL}d {NUM_HEADS}h  d_ff={D_FF}")
    print(f"  context       : {CONTEXT_LENGTH}   vocab: {BPE_VOCAB_SIZE}")
    print(f"  fixed lr      : {LEARNING_RATE:.2e}")
    print(f"  iterations    : {MAX_ITERS}")
    print(f"  batch sizes   : {BATCH_SIZES}")
    print("=" * 70)

    # Load data once (memory-mapped, so basically free)
    print("\nLoading data …")
    train_data = load_memmap(DATA_TRAIN)
    val_data   = load_memmap(DATA_VAL)
    print(f"  train tokens : {len(train_data):,}")
    print(f"  val   tokens : {len(val_data):,}")

    results = []
    for bs in BATCH_SIZES:
        result = run_one_batch_size(bs, train_data, val_data)
        results.append(result)

    # ── Summary table ──────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  BATCH SIZE SWEEP SUMMARY  (lr = {:.2e})".format(LEARNING_RATE))
    print("=" * 70)
    print(f"  {'Batch':>8}  {'Train Loss':>12}  {'Val Loss':>12}  {'Status':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")
    for r in results:
        if r["oom"]:
            status = "OOM"
            tl = vl = "  N/A"
        elif r["diverged"]:
            status = "DIVERGED"
            tl = vl = "  N/A"
        else:
            status = "OK"
            tl = f"{r['final_train_loss']:.4f}"
            vl = f"{r['final_val_loss']:.4f}"
        print(f"  {r['batch_size']:>8d}  {tl:>12}  {vl:>12}  {status:>12}")
    print("=" * 70)

    # Best (non-diverged, non-OOM) run
    valid = [r for r in results if not r["diverged"] and not r["oom"]]
    if valid:
        best = min(valid, key=lambda r: r["final_val_loss"])
        print(
            f"\n  Best batch size: {best['batch_size']}  "
            f"(val loss = {best['final_val_loss']:.4f})"
        )

    tee.close()
    sys.stdout = tee._stdout


if __name__ == "__main__":
    main()
