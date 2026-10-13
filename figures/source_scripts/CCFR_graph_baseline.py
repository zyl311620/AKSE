"""Aligned baseline data for the concept-confusion comparison.

The model order is the canonical order used by CCFR_graph_AKSE.py.
The original baseline script contained additional models that were not
evaluated under AKSE; those models are intentionally excluded here.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "Qwen1.5-7B-Chat",
    "Qwen1.5-14B-Chat",
    "Qwen1.5-32B-Chat",
    "Qwen1.5-72B-Chat",
    "Baichuan2-7B-Chat",
    "Baichuan2-13B-Chat",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-32B",
    "Llama3-8B",
    "Llama3-70B",
    "ChatGLM3-6B",
]

# Values from the original baseline script, restricted to MODEL_ORDER.
BASELINE_DATA = {
    "Qwen1.5-7B-Chat": 88.20,
    "Qwen1.5-14B-Chat": 80.90,
    "Qwen1.5-32B-Chat": 79.84,
    "Qwen1.5-72B-Chat": 58.62,
    "Baichuan2-7B-Chat": 96.72,
    "Baichuan2-13B-Chat": 87.65,
    "DeepSeek-R1-Distill-Llama-8B": 56.43,
    "DeepSeek-R1-Distill-Qwen-14B": 52.41,
    "DeepSeek-R1-Distill-Qwen-32B": 42.04,
    "Llama3-8B": 47.92,
    "Llama3-70B": 40.91,
    "ChatGLM3-6B": 69.45,
}


def get_baseline_dataframe() -> pd.DataFrame:
    """Return baseline values in the canonical AKSE model order."""
    return pd.DataFrame(
        {
            "Model": MODEL_ORDER,
            "Baseline": [BASELINE_DATA[model] for model in MODEL_ORDER],
        }
    )


def plot_aligned_baseline(output_name: str = "CFRR_baseline_aligned.pdf") -> Path:
    """Save an aligned baseline-only chart for data inspection."""
    output_path = Path(__file__).resolve().parent / output_name
    df = get_baseline_dataframe()
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(y, df["Baseline"], color="#9ecae1", label="Baseline")
    ax.set_yticks(y, labels=df["Model"])
    ax.invert_yaxis()
    ax.set_xlabel("Concept Confusion Rate (%)")
    ax.set_title("Concept Confusion Rate - Aligned Baseline")
    ax.set_xlim(0, max(df["Baseline"]) * 1.12)
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(f"Saved: {plot_aligned_baseline()}")
