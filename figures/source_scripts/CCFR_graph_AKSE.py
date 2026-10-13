"""Baseline-versus-AKSE concept-confusion comparison.

The lower the concept-confusion rate, the better the result. The paired
horizontal bars use exactly the same 12-model order as the AKSE experiments.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CCFR_graph_baseline import BASELINE_DATA, MODEL_ORDER


AKSE_DATA = {
    "Qwen1.5-7B-Chat": 73.45,
    "Qwen1.5-14B-Chat": 71.68,
    "Qwen1.5-32B-Chat": 68.43,
    "Qwen1.5-72B-Chat": 54.31,
    "Baichuan2-7B-Chat": 84.36,
    "Baichuan2-13B-Chat": 72.49,
    "DeepSeek-R1-Distill-Llama-8B": 43.68,
    "DeepSeek-R1-Distill-Qwen-14B": 41.05,
    "DeepSeek-R1-Distill-Qwen-32B": 35.45,
    "Llama3-8B": 40.44,
    "Llama3-70B": 31.74,
    "ChatGLM3-6B": 65.49,
}


def get_comparison_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": MODEL_ORDER,
            "Baseline": [BASELINE_DATA[model] for model in MODEL_ORDER],
            "AKSE": [AKSE_DATA[model] for model in MODEL_ORDER],
        }
    )


def plot_comparison(output_name: str = "CFRR_baseline_vs_AKSE.pdf") -> Path:
    output_path = Path(__file__).resolve().parent / output_name
    df = get_comparison_dataframe()
    y = np.arange(len(df))
    bar_height = 0.36

    fig, ax = plt.subplots(figsize=(12, 7.5))
    baseline_bars = ax.barh(
        y - bar_height / 2,
        df["Baseline"],
        height=bar_height,
        color="#c9b6e4",
        edgecolor="#6a3d9a",
        linewidth=0.8,
        label="Baseline (without AKSE)",
    )
    akse_bars = ax.barh(
        y + bar_height / 2,
        df["AKSE"],
        height=bar_height,
        color="#f4a261",
        edgecolor="#b85c00",
        linewidth=0.8,
        label="AKSE",
    )

    ax.set_yticks(y, labels=df["Model"])
    ax.invert_yaxis()
    ax.set_xlabel("Concept Confusion Rate (%)")
    ax.set_title(
        "Concept Confusion Rate: Baseline vs AKSE",
        fontsize=15,
        fontweight="bold",
        color="#5a2a82",
    )
    ax.set_xlim(0, max(df[["Baseline", "AKSE"]].max()) * 1.14)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.bar_label(baseline_bars, fmt="%.2f", padding=3, fontsize=8, color="#5a2a82")
    ax.bar_label(akse_bars, fmt="%.2f", padding=3, fontsize=8, color="#8a3f00")

    for index, row in df.iterrows():
        delta = row["AKSE"] - row["Baseline"]
        ax.text(
            max(row["Baseline"], row["AKSE"]) + 5.0,
            index,
            f"Δ {delta:+.2f}",
            va="center",
            fontsize=8,
            color="#2a9d8f" if delta < 0 else "#d1495b",
            fontweight="bold",
        )

    ax.text(
        0.99,
        0.02,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#5a2a82",
        style="italic",
    )
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(f"Saved: {plot_comparison()}")
