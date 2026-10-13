"""Baseline-versus-AKSE refusal-rate comparisons.

Running this script creates two comparison figures:
  1. RR_baseline_vs_AKSE.pdf/.png
  2. CRR_baseline_vs_AKSE.pdf/.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Refusal_graph_baseline import BASELINE_DATA, MODEL_ORDER


AKSE_DATA = {
    "Qwen1.5-7B-Chat": {
        "Refusal Rate (%)": 11.29,
        "Correct Refusal Rate (%)": 73.15,
    },
    "Qwen1.5-14B-Chat": {
        "Refusal Rate (%)": 12.82,
        "Correct Refusal Rate (%)": 92.36,
    },
    "Qwen1.5-32B-Chat": {
        "Refusal Rate (%)": 11.35,
        "Correct Refusal Rate (%)": 68.62,
    },
    "Qwen1.5-72B-Chat": {
        "Refusal Rate (%)": 11.82,
        "Correct Refusal Rate (%)": 79.80,
    },
    "Baichuan2-7B-Chat": {
        "Refusal Rate (%)": 9.36,
        "Correct Refusal Rate (%)": 84.06,
    },
    "Baichuan2-13B-Chat": {
        "Refusal Rate (%)": 16.72,
        "Correct Refusal Rate (%)": 95.56,
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "Refusal Rate (%)": 13.31,
        "Correct Refusal Rate (%)": 72.62,
    },
    "DeepSeek-R1-Distill-Qwen-14B": {
        "Refusal Rate (%)": 16.58,
        "Correct Refusal Rate (%)": 81.43,
    },
    "DeepSeek-R1-Distill-Qwen-32B": {
        "Refusal Rate (%)": 18.63,
        "Correct Refusal Rate (%)": 86.20,
    },
    "Llama3-8B": {
        "Refusal Rate (%)": 14.63,
        "Correct Refusal Rate (%)": 92.11,
    },
    "Llama3-70B": {
        "Refusal Rate (%)": 30.49,
        "Correct Refusal Rate (%)": 93.05,
    },
    "ChatGLM3-6B": {
        "Refusal Rate (%)": 16.80,
        "Correct Refusal Rate (%)": 40.28,
    },
}


def get_comparison_dataframe() -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "Model": model,
                "Baseline Refusal Rate (%)": BASELINE_DATA[model][
                    "Refusal Rate (%)"
                ],
                "AKSE Refusal Rate (%)": AKSE_DATA[model]["Refusal Rate (%)"],
                "Baseline Correct Refusal Rate (%)": BASELINE_DATA[model][
                    "Correct Refusal Rate (%)"
                ],
                "AKSE Correct Refusal Rate (%)": AKSE_DATA[model][
                    "Correct Refusal Rate (%)"
                ],
            }
        )
    return pd.DataFrame(rows)


def plot_metric(
    df: pd.DataFrame,
    baseline_column: str,
    akse_column: str,
    ylabel: str,
    title: str,
    output_stem: str,
) -> Path:
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / f"{output_stem}.pdf"
    x = np.arange(len(df))
    bar_width = 0.38

    if "Correct" in ylabel:
        baseline_color = "#b7b7b7"
        akse_color = "#2ca25f"
        baseline_edge = "#636363"
        akse_edge = "#006d2c"
        baseline_hatch = "///"
        title_color = "#176b3a"
        note = "Higher is better"
    else:
        baseline_color = "#9ecae1"
        akse_color = "#de2d26"
        baseline_edge = "#3182bd"
        akse_edge = "#a50f15"
        baseline_hatch = "//"
        title_color = "#9c1c1c"
        note = "Interpret together with correct refusal rate"

    fig, ax = plt.subplots(figsize=(13, 7.5))
    baseline_bars = ax.bar(
        x - bar_width / 2,
        df[baseline_column],
        width=bar_width,
        color=baseline_color,
        edgecolor=baseline_edge,
        linewidth=0.8,
        hatch=baseline_hatch,
        label="Baseline (without AKSE)",
    )
    akse_bars = ax.bar(
        x + bar_width / 2,
        df[akse_column],
        width=bar_width,
        color=akse_color,
        edgecolor=akse_edge,
        linewidth=0.8,
        label="AKSE",
    )

    ax.set_xticks(x, labels=df["Model"], rotation=55, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=15, fontweight="bold", color=title_color)
    max_value = max(df[[baseline_column, akse_column]].max())
    ax.set_ylim(0, max_value * 1.20)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.bar_label(baseline_bars, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(akse_bars, fmt="%.2f", padding=3, fontsize=8)

    for index, row in df.iterrows():
        delta = row[akse_column] - row[baseline_column]
        ax.text(
            index,
            max(row[baseline_column], row[akse_column]) + max_value * 0.085,
            f"Δ {delta:+.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#006d2c" if (("Correct" in ylabel and delta > 0) or
                                ("Correct" not in ylabel and delta < 0))
            else "#a50f15",
            fontweight="bold",
        )

    ax.text(
        0.99,
        0.98,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=title_color,
        style="italic",
    )
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_comparisons() -> list[Path]:
    df = get_comparison_dataframe()
    return [
        plot_metric(
            df,
            "Baseline Refusal Rate (%)",
            "AKSE Refusal Rate (%)",
            "Refusal Rate (%)",
            "Refusal Rate: Baseline vs AKSE",
            "RR_baseline_vs_AKSE",
        ),
        plot_metric(
            df,
            "Baseline Correct Refusal Rate (%)",
            "AKSE Correct Refusal Rate (%)",
            "Correct Refusal Rate (%)",
            "Correct Refusal Rate: Baseline vs AKSE",
            "CRR_baseline_vs_AKSE",
        ),
    ]


if __name__ == "__main__":
    for path in plot_comparisons():
        print(f"Saved: {path}")
