"""Create the paper's small, reproducible summary figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "pdf.fonttype": 42,
})


def pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    ax.set_xlim(0, 7); ax.set_ylim(0, 2.85); ax.axis("off")

    navy = "#1f3b53"
    blue = "#356f9f"
    ink = "#27343b"
    muted = "#5d6b73"
    panels = ["#f4f6f7", "#eaf2f7", "#f4f1e9"]

    def panel(x, y, w, h, title, number, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color,
                               edgecolor="#87939a", linewidth=.8, linestyle=(0, (3, 2))))
        ax.text(x + .10, y + h + .10, f"{number}. {title}", ha="left", va="bottom",
                fontsize=7.7, weight="bold", color=ink)

    def box(x, y, w, h, title, subtitle="", fill="#ffffff", edge=navy, size=6.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.025,rounding_size=.035",
                                    facecolor=fill, edgecolor=edge, linewidth=.8))
        ax.text(x + w / 2, y + h * .62, title, ha="center", va="center",
                fontsize=size, weight="bold", color=ink, wrap=True)
        if subtitle:
            ax.text(x + w / 2, y + h * .25, subtitle, ha="center", va="center",
                    fontsize=5.7, color=muted)

    def arrow(x1, y1, x2, y2, color=blue, width=.85, scale=7):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=scale, linewidth=width, color=color))

    panel(.08, .80, 2.05, 1.62, "Evidence construction", "1", panels[0])
    box(.22, 1.79, .78, .36, "PDF corpus", "papers + reports", fill="#ffffff")
    box(1.20, 1.79, .78, .36, "Question", "unit + task type", fill="#ffffff")
    arrow(.61, 1.76, .61, 1.55, muted, .7, 6)
    arrow(1.59, 1.76, 1.59, 1.55, muted, .7, 6)
    box(.40, 1.05, 1.42, .40, "BM25 + dense retrieval", "candidate passages", fill="#e8f1f5", edge=blue)
    arrow(1.05, 1.02, 1.05, .88, blue, .8, 7)

    panel(2.43, .80, 2.05, 1.62, "Evidence control", "2", panels[1])
    box(2.64, 1.79, .75, .36, "RRF fusion", "rank lists", fill="#ffffff")
    box(3.52, 1.79, .75, .36, "Reranker", "question-passage", fill="#ffffff")
    arrow(3.40, 1.97, 3.50, 1.97, blue, .8, 7)
    box(2.88, 1.05, 1.18, .40, "Top-k evidence", "small grounded context", fill="#dcebf4", edge=blue)
    arrow(3.02, 1.76, 3.32, 1.48, blue, .8, 7)
    arrow(3.89, 1.76, 3.62, 1.48, blue, .8, 7)

    panel(4.78, .80, 2.14, 1.62, "Generation and checks", "3", panels[2])
    box(4.98, 1.79, .78, .36, "Generator", "structured JSON", fill="#ffffff")
    box(5.92, 1.79, .78, .36, "Validator", "value + unit + refs", fill="#ffffff")
    arrow(5.78, 1.97, 5.88, 1.97, blue, .8, 7)
    box(5.05, 1.05, 1.45, .40, "repair | normalize | abstain", "checked prediction", fill="#e7f0df", edge="#4f704d", size=6.2)
    arrow(5.37, 1.76, 5.64, 1.48, blue, .8, 7)
    arrow(6.31, 1.76, 6.00, 1.48, blue, .8, 7)

    ax.text(3.50, .45, "Evaluation contract", ha="center", va="center", fontsize=7.3,
            weight="bold", color=ink)
    for x, title, sub in [(1.05, "WattBot 2026", "blind Kaggle score"),
                          (3.50, "WattBot 2025", "41-question transfer"),
                          (5.95, "SciFact", "retrieval transfer")]:
        box(x - .68, .05, 1.36, .28, title, sub, fill="#ffffff", edge="#87939a", size=6.1)
        arrow(x, .78, x, .36, muted, .7, 6)

    ax.text(3.50, 2.73, "EvidenceFirst: retrieve, verify, then reason", ha="center",
            va="center", fontsize=9.1, weight="bold", color=navy)
    fig.tight_layout(pad=.12)
    fig.savefig(OUT / "pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def retrieval():
    datasets = ["WattBot 2025", "SciFact"]
    bm25 = [75.61, 70.20]
    hybrid = [66.67, 68.28]
    reranked = [80.77, 82.77]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    x = [0, 1]
    series = [(bm25, "BM25", "#9aa6ad", "o"), (hybrid, "Hybrid", "#4f6d7a", "s"),
              (reranked, "Hybrid + reranker", "#1b4f72", "D")]
    for values, label, color, marker in series:
        ax.plot(x, values, color=color, marker=marker, markersize=6, linewidth=1.4, label=label)
        for i, v in enumerate(values):
            ax.annotate(f"{v:.1f}", (x[i], v), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, color=color)
    ax.set_xticks(x, datasets); ax.set_ylim(60, 90); ax.set_ylabel("Recall@1 (%)")
    ax.set_title("Early evidence recall across transfer sets", pad=8)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(.5, -0.28))
    ax.grid(axis="y", color="#d9dee1", linewidth=.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(OUT / "retrieval_r1.pdf", bbox_inches="tight")
    plt.close(fig)


def normalization():
    labels = ["Original", "Range/unit\nnormalization", "+ reciprocal\nconversion"]
    values = [.731, .767, .786]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    x = list(range(len(values)))
    ax.plot(x, values, color="#1b4f72", linewidth=1.8, marker="o", markersize=6)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3f}", (i, v), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, weight="bold", color="#1b4f72")
    ax.annotate("+0.055", xy=(2, values[2]), xytext=(1.45, .805),
                arrowprops=dict(arrowstyle="-|>", color="#50616d", linewidth=.8),
                fontsize=7.5, color="#50616d")
    ax.set_xticks(x, labels); ax.set_ylim(.70, .81); ax.set_ylabel("Overall score")
    ax.set_title("Conservative normalization improves transfer accuracy", pad=8)
    ax.grid(axis="y", color="#d9dee1", linewidth=.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "normalization.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pipeline(); retrieval(); normalization()
