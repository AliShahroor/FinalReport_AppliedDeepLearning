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
    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    ax.set_xlim(0, 7); ax.set_ylim(0, 3.15); ax.axis("off")

    navy = "#1f3b53"
    blue = "#356f9f"
    ink = "#27343b"
    muted = "#5d6b73"

    def panel(x, y, w, h, title, number, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color,
                               edgecolor="#87939a", linewidth=.8, linestyle=(0, (3, 2))))
        ax.text(x + .10, y + h + .08, f"{number}. {title}", ha="left", va="bottom",
                fontsize=7.5, weight="bold", color=ink)

    def box(x, y, w, h, title, subtitle="", fill="#ffffff", edge=navy, size=6.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.025,rounding_size=.035",
                                    facecolor=fill, edgecolor=edge, linewidth=.8))
        ax.text(x + w / 2, y + h * .63, title, ha="center", va="center",
                fontsize=size, weight="bold", color=ink)
        if subtitle:
            ax.text(x + w / 2, y + h * .24, subtitle, ha="center", va="center",
                    fontsize=5.5, color=muted)

    def arrow(x1, y1, x2, y2, color=blue, width=.8, scale=7):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=scale, linewidth=width, color=color))

    # The top row is the actual method; the bottom row is the evaluation contract.
    panel(.08, 1.25, 1.55, 1.42, "Inputs", "1", "#f4f6f7")
    box(.22, 2.00, .58, .34, "PDFs", "source corpus")
    box(.91, 2.00, .58, .34, "Question", "unit + type")
    box(.39, 1.43, .93, .34, "query + corpus", "retrieval input", fill="#e8f1f5", edge=blue, size=6.0)
    arrow(.51, 1.97, .69, 1.79, muted, .7, 6)
    arrow(1.20, 1.97, 1.02, 1.79, muted, .7, 6)

    panel(1.82, 1.25, 2.35, 1.42, "Retrieval and evidence control", "2", "#eaf2f7")
    box(2.00, 2.00, .80, .34, "BM25", "lexical rank", fill="#ffffff")
    box(3.15, 2.00, .80, .34, "Dense", "semantic rank", fill="#ffffff")
    box(2.52, 1.56, .92, .34, "RRF fusion", "combine ranks", fill="#ffffff")
    box(2.43, 1.02, 1.10, .34, "Reranker", "question-passage", fill="#dcebf4", edge=blue)
    arrow(2.40, 1.97, 2.78, 1.92, blue, .75, 6)
    arrow(3.55, 1.97, 3.18, 1.92, blue, .75, 6)
    arrow(2.98, 1.54, 2.98, 1.39, blue, .75, 6)
    arrow(1.64, 1.70, 2.35, 2.00, blue, .75, 6)
    arrow(1.64, 1.54, 3.50, 2.00, blue, .75, 6)

    panel(4.36, 1.25, 2.56, 1.42, "Generation and validation", "3", "#f4f1e9")
    box(4.55, 2.00, .92, .34, "Generator", "structured JSON", fill="#ffffff")
    box(5.82, 2.00, .88, .34, "Validator", "schema + refs", fill="#ffffff")
    box(4.92, 1.50, 1.48, .34, "repair / normalize", "format + units", fill="#ffffff", edge="#4f704d", size=6.1)
    box(5.08, 1.02, 1.17, .34, "checked output", "or abstain", fill="#e7f0df", edge="#4f704d")
    arrow(5.47, 2.00, 5.78, 2.00, blue, .75, 6)
    arrow(6.26, 1.98, 5.91, 1.84, blue, .75, 6)
    arrow(5.66, 1.48, 5.66, 1.38, muted, .7, 6)
    arrow(5.66, 1.00, 5.66, .84, blue, .75, 6)
    arrow(3.54, 1.19, 4.48, 1.19, blue, .8, 7)

    ax.text(3.50, .63, "Evaluation contract", ha="center", va="center", fontsize=7.2,
            weight="bold", color=ink)
    for x, title, sub in [(1.05, "WattBot 2026", "blind Kaggle score"),
                          (3.50, "WattBot 2025", "41-question transfer"),
                          (5.95, "SciFact", "retrieval transfer")]:
        box(x - .68, .17, 1.36, .30, title, sub, fill="#ffffff", edge="#87939a", size=6.0)
        arrow(5.66, .99, x, .49, muted, .65, 6)

    ax.text(3.50, 2.99, "EvidenceFirst: retrieve, verify, then reason", ha="center",
            va="center", fontsize=9.0, weight="bold", color=navy)
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
