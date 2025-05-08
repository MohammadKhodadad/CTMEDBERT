import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── 0. paths ───────────────────────────────────────────────────────────────────
log_file = Path("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/slurm-43071583.out")

# ── 1. pull rolling training losses ────────────────────────────────────────────
train_pat = re.compile(
    r"Epoch\s+(\d+)\s+\(Training\).*?\|\s*(\d+)/\d+.*?Step=(\d+),\s*Loss=([\d.]+)"
)
eval_pat = re.compile(
    r"Epoch\s+(\d+),\s+Evaluation Loss:\s+([\d.]+)"
)

train_rows, seen = [], set()        # dedupe duplicated bar lines
eval_rows  = []

with log_file.open() as f:
    for ln in f:
        m = train_pat.search(ln)
        if m:
            epoch, n_seen, step, avg = int(m[1]), int(m[2]) + 1, int(m[3]), float(m[4])
            if (epoch, step) not in seen:          # progress-bar prints twice
                seen.add((epoch, step))
                train_rows.append((epoch, n_seen, step, avg))
            continue

        m = eval_pat.search(ln)
        if m:
            epoch, e_loss = int(m[1]), float(m[2])
            eval_rows.append((epoch, e_loss))

# DataFrame of rolling data -----------------------------------------------------
df = pd.DataFrame(train_rows, columns=["epoch", "n_in_epoch", "step", "avg_loss"])
df.sort_values(["epoch", "n_in_epoch"], inplace=True, ignore_index=True)

# ── 2. rolling → true per-step loss -------------------------------------------
true_losses = []

prev_epoch   = None
cumul_prev   = 0.0          # Σ loss up to previous row *within the epoch*
n_prev       = 0            # samples up to previous row  (same)

for epoch, n_seen, step, avg in df[["epoch", "n_in_epoch", "step", "avg_loss"]].itertuples(index=False):
    # ── reset when a new epoch begins ─────────────────────────────────────────
    if epoch != prev_epoch:
        cumul_prev = 0.0
        n_prev     = 0

    cumul_now = avg * n_seen              # Σ loss up to this step
    true      = cumul_now - cumul_prev    # loss of *this* batch

    true_losses.append(true)

    # update trackers for next iteration
    cumul_prev, n_prev, prev_epoch = cumul_now, n_seen, epoch

df["true_loss"] = true_losses

# ── 3. 512-step means ----------------------------------------------------------
df_512 = (
    df
    .assign(bin=lambda x: x["step"] // 512)
    .groupby("bin", as_index=False)
    .agg({"step": "last", "true_loss": "mean"})   # step = right edge of bin
    .rename(columns={"true_loss": "loss_512_mean"})
)

# ── 4. map eval loss → global step --------------------------------------------
#      (evaluation printed just after last train step of that epoch)
epoch_last_step = df.groupby("epoch")["step"].max()
eval_points = [
    (epoch_last_step[e], loss) for e, loss in eval_rows if e in epoch_last_step
]
eval_steps, eval_losses = zip(*eval_points) if eval_points else ([], [])

# ── 5. plot --------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ensure eval points are in ascending-step order before connecting them
if eval_steps:
    eval_order = np.argsort(eval_steps)
    eval_steps_sorted  = np.array(eval_steps)[eval_order]
    eval_losses_sorted = np.array(eval_losses)[eval_order]

plt.figure(figsize=(10, 5))

# Training curve – blue
plt.plot(
    df_512["step"],
    df_512["loss_512_mean"],
    color="tab:blue",
    label="Train (mean of 512 steps)"
)

# Evaluation curve – orange, connected
if eval_steps:
    plt.plot(
        eval_steps_sorted,
        eval_losses_sorted,
        color="tab:orange",
        marker="o",
        linestyle="-",
        label="Eval loss"
    )

plt.xlabel("Global training step")
plt.ylabel("Loss")
plt.title("Learning Curve: Training (512-Step Mean) and Evaluation Loss vs. Global Step")
plt.legend()
plt.tight_layout()
plt.savefig("loss.jpg", format="jpg", dpi=300)
plt.savefig("loss.pdf", format="pdf", dpi=300)
plt.show()