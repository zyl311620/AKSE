"""Aligned baseline data for refusal-rate comparisons.

Only models that also have AKSE results are retained. The canonical model
order is shared with Refusal_graph_AKSE.py.
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
    "Qwen1.5-7B-Chat": {
        "Refusal Rate (%)": 8.98,
        "Correct Refusal Rate (%)": 31.77,
    },
    "Qwen1.5-14B-Chat": {
        "Refusal Rate (%)": 10.29,
        "Correct Refusal Rate (%)": 39.16,
    },
    "Qwen1.5-32B-Chat": {
        "Refusal Rate (%)": 10.34,
        "Correct Refusal Rate (%)": 65.27,
    },
    "Qwen1.5-72B-Chat": {
        "Refusal Rate (%)": 13.97,
        "Correct Refusal Rate (%)": 79.31,
    },
    "Baichuan2-7B-Chat": {
        "Refusal Rate (%)": 1.19,
        "Correct Refusal Rate (%)": 3.44,
    },
    "Baichuan2-13B-Chat": {
        "Refusal Rate (%)": 6.63,
        "Correct Refusal Rate (%)": 25.46,
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "Refusal Rate (%)": 8.86,
        "Correct Refusal Rate (%)": 66.01,
    },
    "DeepSeek-R1-Distill-Qwen-14B": {
        "Refusal Rate (%)": 10.54,
        "Correct Refusal Rate (%)": 69.70,
    },
    "DeepSeek-R1-Distill-Qwen-32B": {
        "Refusal Rate (%)": 11.42,
        "Correct Refusal Rate (%)": 74.87,
    },
    "Llama3-8B": {
        "Refusal Rate (%)": 7.68,
        "Correct Refusal Rate (%)": 51.23,
    },
    "Llama3-70B": {
        "Refusal Rate (%)": 24.69,
        "Correct Refusal Rate (%)": 56.84,
    },
    "ChatGLM3-6B": {
        "Refusal Rate (%)": 14.70,
        "Correct Refusal Rate (%)": 26.35,
    },
}


def get_baseline_dataframe() -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        rows.append({"Model": model, **BASELINE_DATA[model]})
    return pd.DataFrame(rows)


def plot_aligned_baselines() -> list[Path]:
    """Save aligned baseline-only charts for data inspection."""
    output_dir = Path(__file__).resolve().parent
    df = get_baseline_dataframe()
    outputs = []
    y = np.arange(len(df))

    for metric, color, filename, title in [
        (
            "Refusal Rate (%)",
            "#9ecae1",
            "RR_baseline_aligned.pdf",
            "Refusal Rate - Aligned Baseline",
        ),
        (
            "Correct Refusal Rate (%)",
            "#74c476",
            "CRR_baseline_aligned.pdf",
            "Correct Refusal Rate - Aligned Baseline",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(11, 7))
        bars = ax.barh(y, df[metric], color=color, label="Baseline")
        ax.set_yticks(y, labels=df["Model"])
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_title(title)
        ax.set_xlim(0, max(df[metric]) * 1.14)
        ax.grid(axis="x", alpha=0.25)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)
    return outputs


if __name__ == "__main__":
    for path in plot_aligned_baselines():
        print(f"Saved: {path}")
