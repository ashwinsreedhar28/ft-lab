"""Curves from runs/<name>/log_history.json.

    python -m ftlab.plot runs/gsm8k_h100 [runs/gsm8k_lora ...]

Writes <first run>/curves.png: reward (raw + rolling mean), per-reward means,
KL, completion length, step time. Several runs overlay on the same axes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#0891b2"]


def _series(hist, key):
    xs, ys = [], []
    for h in hist:
        if key in h and "step" in h:
            xs.append(h["step"])
            ys.append(h[key])
    return xs, ys


def _roll(ys, w=10):
    out = []
    for i in range(len(ys)):
        win = ys[max(0, i - w + 1): i + 1]
        out.append(sum(win) / len(win))
    return out


def main(run_dirs: list[str]):
    runs = [(Path(d).name, json.loads((Path(d) / "log_history.json").read_text())) for d in run_dirs]
    reward_keys = sorted({k for _, h in runs for e in h for k in e if k.startswith("rewards/") and k.endswith("/mean")})
    panels = [("reward", "reward (sum)"), *[(k, k.split("/")[1]) for k in reward_keys],
              ("kl", "KL to reference"), ("completions/mean_length", "completion length (tok)"),
              ("completions/clipped_ratio", "clipped (hit max len)"), ("step_time", "step time (s)")]
    n = len(panels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, panels):
        for (name, hist), color in zip(runs, PALETTE):
            xs, ys = _series(hist, key)
            if not xs:
                continue
            ax.plot(xs, ys, color=color, alpha=0.25, linewidth=1)
            ax.plot(xs, _roll(ys), color=color, linewidth=2, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.2)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = Path(run_dirs[0]) / "curves.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
