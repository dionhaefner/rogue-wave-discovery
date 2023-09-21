import sys
import json
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import patheffects
import matplotlib.pyplot as plt

import aquarel
theme = (
    aquarel.load_theme("arctic_light")
    .set_grid(draw=True, width=0.5)
    .set_font(family="sans-serif", sans_serif=["Source Sans Pro"])
    .set_color(grid_color="white", text_color="black", plot_background_color="0.95")
)
theme.apply()


def generate_plots(datafile):
    with open(datafile, "r") as f:
        data = json.load(f)

    # sort by total number of parameter interactions
    num_interactions = lambda n: 0.5 * (n - 1)**2 + n
    data = sorted(data, key=lambda d: sum(num_interactions(len(k)) for k in d["feature_groups"]), reverse=False)

    data_transposed = defaultdict(list)

    for row in data:
        for score, scoreval in row["scores"].items():
            data_transposed[score].append(scoreval)

    data_transposed = {k: np.array(v) for k, v in data_transposed.items()}
    model_rank = np.empty(len(data))
    model_rank[np.argsort(data_transposed["test_score_subsets"])[::-1]] = np.arange(len(data))

    score = data_transposed["consistency_score"]
    err = data_transposed["consistency_err"]

    pareto_idx = []
    best_err = np.inf
    for i in np.argsort(score)[::-1]:
        if err[i] < best_err:
            pareto_idx.append(i)
            best_err = err[i]
        
    cutoff = round(0.2 * len(data))
    mask_good = model_rank < cutoff
    mask_bad = model_rank >= len(data) - cutoff
    mask_meh = np.logical_not(np.logical_or(mask_good, mask_bad))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5))

    plt.step(score[pareto_idx], err[pareto_idx], color="0.1", linewidth=1.2, where="post", linestyle="dashed")
    plt.annotate("Pareto frontier".upper(), font="monospace", xy=(0.5 * (score[pareto_idx[-1]] + score[pareto_idx[-2]]), err[pareto_idx[-2]]), xytext=(0, -5), textcoords="offset points", size=11, color="k", ha="center", va="top")

    plt.scatter(score[mask_good], err[mask_good], marker="p", facecolors="#fed667", edgecolors="0.1", zorder=10, label=f"Top {cutoff} models", sizes=[100])
    plt.scatter(score[mask_meh], err[mask_meh], marker="p", facecolors="white", edgecolors="0.1", zorder=9, label=f"Middle {len(data) - 2 * cutoff} models", sizes=[100])
    plt.scatter(score[mask_bad], err[mask_bad], marker="p", facecolors="0.3", edgecolors="0.1", zorder=10, label=f"Bottom {cutoff} models", sizes=[100])

    path_effects = [patheffects.withStroke(linewidth=1.5, foreground="w")]

    for model in np.where(mask_good)[0]:
        if model == 15:
            # prevent overlap with model 16
            continue

        plt.annotate(str(model + 1), xy=(score[model], err[model]), xytext=(0, -10), font="monospace", textcoords="offset points", size=9, ha="center", va="center", color="black", zorder=10, path_effects=path_effects)

    for model in np.where(mask_bad)[0]:
        plt.annotate(str(model + 1), xy=(score[model], err[model]), xytext=(0, -10), font="monospace", textcoords="offset points", size=9, ha="center", va="center", color="black", zorder=10, path_effects=path_effects)

    chosen_model = 17
    plt.annotate("Chosen model", xy=(score[chosen_model] + 0.05e-4, err[chosen_model] - 0.05e-1), xytext=(30, -15), font="monospace", textcoords="offset points", size=11, color="black", ha="center", va="top", arrowprops=dict(arrowstyle="->", color="0.2", linewidth=1))

    plt.xlabel("Prediction score $\\mathcal{L}$")
    plt.ylabel("Causal invariance error $\\mathcal{E}$")
    plt.xlim(3.6e-4, 7e-4)
    plt.ylim(0.07, 0.22)
    plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(title="Test performance", facecolor="white", frameon=True, markerscale=1, title_fontproperties=dict(weight="semibold", size=12), fontsize=11, handletextpad=0.1)
    fig.tight_layout()
    plt.savefig("generated/pareto.pdf", bbox_inches="tight")


if __name__ == "__main__":
    generate_plots(sys.argv[1])