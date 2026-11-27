"""Render each scientific panel from audited predictions or deposited coordinates."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from settings import BUDGETS, DATA, FEATURE_SETS, MODELS, RESULTS


TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
SPLIT_NAMES = {"random": "Random variants", "position": "Held-out residue positions"}
COLORS = {"Conventional": "#087F74", "SaProt": "#A54828", "Both": "#315B9F",
          "buried": "#3E566F", "exposed": "#AA731C", "Observed": "#263440"}
SHORT = {"Conventional": "+ Conventional structure", "SaProt": "+ SaProt", "Both": "+ Both"}


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "white",
                         "axes.labelcolor": "#202F3D", "axes.titlecolor": "#202F3D",
                         "text.color": "#202F3D", "xtick.color": "#32465A", "ytick.color": "#32465A",
                         "axes.edgecolor": "#9BACB8", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=260, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def overall_plot(performance: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.1, 4.4), layout="constrained", sharey=True)
    colors = LinearSegmentedColormap.from_list("accuracy", ["#f3f7fa", "#9ec7c1", "#17665e", "#123d3a"])
    for ax, split in zip(axes, SPLIT_NAMES, strict=True):
        data = performance[(performance["split"] == split) & (performance["train_n"] == 500)]
        table = data.pivot(index="features", columns="model", values="mean_rho").loc[list(FEATURE_SETS), list(MODELS)]
        image = ax.imshow(table.to_numpy(), cmap=colors, vmin=0.15, vmax=0.56, aspect="auto")
        ax.set_title(SPLIT_NAMES[split], fontweight="bold", pad=10)
        ax.set_xticks(np.arange(len(MODELS)), MODELS)
        ax.set_yticks(np.arange(len(FEATURE_SETS)), FEATURE_SETS)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(FEATURE_SETS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        row_best, col_best = np.unravel_index(np.argmax(table.to_numpy()), table.shape)
        for (i, j), value in np.ndenumerate(table.to_numpy()):
            is_best = (i, j) == (row_best, col_best)
            rgb = np.array(image.to_rgba(value)[:3])
            linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            luminance = linear_rgb @ np.array([0.2126, 0.7152, 0.0722])
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if luminance < 0.225 else "#14283b", fontweight="bold" if is_best else "normal")
        ax.add_patch(plt.Rectangle((col_best - 0.48, row_best - 0.48), 0.96, 0.96,
                                   fill=False, lw=2, ec="#ffba52"))
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = figure.colorbar(image, ax=axes, fraction=0.045, shrink=0.92, pad=0.035)
    bar.set_label("Mean test Spearman ρ")
    figure.suptitle("Prediction accuracy at 500 training labels", fontsize=15, fontweight="bold")
    save(figure, "fig01_performance")


def learning_plot(paired: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 5.4), sharex=True, sharey=True, layout="constrained")
    for row, split in enumerate(SPLIT_NAMES):
        for column, model in enumerate(MODELS):
            ax = axes[row, column]
            data = paired[(paired["split"] == split) & (paired["model"] == model)
                          & (paired["environment"] == "all")]
            for comparison in SHORT:
                subset = data[data["comparison"] == comparison].set_index("train_n").loc[list(BUDGETS)]
                x = np.arange(len(BUDGETS))
                mean, sd = subset["mean_delta"].to_numpy(), subset["sd_seed_macro"].to_numpy()
                ax.plot(x, mean, color=COLORS[comparison], lw=2.1, marker="o", ms=4.5)
                if comparison != "Both":
                    ax.fill_between(x, mean - sd, mean + sd, color=COLORS[comparison], alpha=0.13, linewidth=0)
            ax.axhline(0, color="#81909d", lw=1, ls="--", zorder=0)
            ax.grid(axis="y", alpha=0.19)
            ax.set_xticks(np.arange(len(BUDGETS)), BUDGETS)
            ax.tick_params(length=0)
            if row == 0:
                ax.set_title(model, fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"{SPLIT_NAMES[split]}\nPaired Δρ")
    handles = [Line2D([0], [0], color=COLORS[key], lw=2, marker="o", label=text) for key, text in SHORT.items()]
    axes[0, 1].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.32), ncol=3, frameon=False)
    figure.supxlabel("Number of training labels", fontsize=10)
    save(figure, "fig02_learning_curves")


def environment_plot(matched: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 2.95), sharey=True, layout="constrained")
    matched = matched[(matched["split"] == "position") & (matched["train_n"] == 500)]
    for ax, model in zip(axes, MODELS, strict=True):
        data = matched[matched["model"] == model]
        for i, key in enumerate(("Conventional", "SaProt")):
            for j, env in enumerate(("buried", "exposed")):
                values = data[(data["comparison"] == key) & (data["environment"] == env)]
                means = values.groupby("assay")["delta_rho"].mean()
                value = means.mean()
                variability = values.groupby("seed")["delta_rho"].mean().std()
                x = i + (j - 0.5) * 0.33
                ax.bar(x, value, width=0.28, color=COLORS[env], zorder=2)
                ax.errorbar(x, value, yerr=variability, color="#25394c", linewidth=0.9, capsize=3, zorder=3)
        ax.set_xticks([0, 1], ["Conventional", "SaProt"])
        ax.axhline(0, color="#81909d", linewidth=0.9)
        ax.set_title(model, fontweight="bold", pad=5)
        ax.grid(axis="y", alpha=0.17)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Paired Δρ")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[group], label=group.capitalize())
               for group in ("buried", "exposed")]
    axes[-1].legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False)
    save(figure, "fig03_environments")


def gap_plot(gap: pd.DataFrame) -> None:
    figure, ax = plt.subplots(figsize=(7.1, 3.6), layout="constrained")
    gap = gap[gap["train_n"] == 500]
    groups = ("D+E", "D+E+S", "D+E+P", "D+E+P+S")
    colors = ("#506987", "#0b8075", "#a27c33")
    for j, (model, color) in enumerate(zip(MODELS, colors, strict=True)):
        part = gap[gap["model"] == model]
        means = part.groupby("features")["rho_gap"].mean().loc[list(groups)]
        variation = part.groupby(["features", "seed"])["rho_gap"].mean().groupby("features").std().loc[list(groups)]
        ax.bar(np.arange(len(groups)) + (j - 1) * 0.23, means, width=0.20, color=color, label=model,
               yerr=variation, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#324253"})
    ax.axhline(0, color="#81909d", lw=0.9)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylabel("ρ(random) − ρ(held-out positions)")
    ax.set_title("Effect of holding out residue positions", fontweight="bold")
    ax.grid(axis="y", alpha=0.19)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    save(figure, "fig04_generalization_gap")


def use_structure(ax, residues: pd.DataFrame, position_colors: dict[int, str]) -> None:
    coords = residues[["x", "y", "z"]].to_numpy(dtype=float)
    coords -= coords.mean(axis=0)
    sites = [(residue, np.flatnonzero(residues["position"].to_numpy() == residue).item(), color)
             for residue, color in position_colors.items()]
    if len(sites) > 1:
        horizontal = coords[sites[-1][1]] - coords[sites[0][1]]
        horizontal /= np.linalg.norm(horizontal)
        orthogonal = coords - np.outer(coords @ horizontal, horizontal)
        _, _, basis = np.linalg.svd(orthogonal, full_matrices=False)
        projection = np.stack([horizontal, basis[0]], axis=1)
    else:
        _, _, basis = np.linalg.svd(coords, full_matrices=False)
        projection = basis[:2].T
    xy = coords @ projection
    ax.plot(*xy.T, c="#71949c", linewidth=1.35, alpha=0.9, zorder=1)
    for index, (residue, i, color) in enumerate(sites):
        ax.scatter(*xy[i], s=58, color=color, edgecolors="white", linewidth=0.7, zorder=4)
        offset = (-26, 24) if index == 0 else (24, -22)
        ax.annotate(str(residue), xy[i], xytext=offset, textcoords="offset points", ha="center", va="center",
                    color=color, fontweight="bold", fontsize=10,
                    bbox={"fc": "white", "ec": "none", "pad": 1.0},
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 1.05}, zorder=5)
    ax.set_aspect("equal")
    ax.margins(0.13)
    ax.axis("off")


def cases_plot(cases: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(9.0, 3.6), layout="constrained")
    n_structures = cases["assay"].nunique()
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.55]) if n_structures == 1 else figure.add_gridspec(1, 3, width_ratios=[1, 1, 1.6])
    for j, (key, subset) in enumerate(cases.groupby("assay", sort=False)):
        axes = figure.add_subplot(grid[0, j])
        residues = pd.read_csv(DATA / "processed" / key / "residues.csv")
        used = pd.read_csv(DATA / "processed" / key / "variants.csv", usecols=["position"])["position"]
        residues = residues[residues["position"].between(used.min(), used.max())]
        position_colors = {row.position: COLORS["Conventional"] if row.error_reduction > 0 else COLORS["SaProt"]
                           for row in subset.itertuples(index=False)}
        use_structure(axes, residues, position_colors)
        axes.set_title(f"{subset['label'].iloc[0]} WT Cα trace", fontweight="bold", pad=4)
    ax = figure.add_subplot(grid[0, -1])
    markers = [("observed_rank", "Observed", "D", COLORS["Observed"]),
               ("baseline_rank", "D + E", "o", COLORS["Both"]),
               ("structural_rank", "D + E + S", "s", COLORS["Conventional"])]
    rows = list(cases.itertuples(index=False))
    for i, case in enumerate(rows):
        ax.axhline(i, color="#d7e0e7", zorder=0)
        for j, (field, label, marker, color) in enumerate(markers):
            ax.scatter(100 * getattr(case, field), i + (j - 1) * 0.13, marker=marker, color=color,
                       edgecolors="white", linewidth=0.8, s=76, label=label if i == 0 else None, zorder=2)
    ax.set_xlim(-5, 105)
    ax.set_ylim(len(rows) - 0.52, -0.63)
    ax.set_yticks(range(len(rows)), [f"{c.mutant}\n({c.environment})" for c in rows])
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_xlabel("Mean percentile rank in outer test set")
    ax.set_title("Opposite effects of adding structure", fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
    save(figure, "fig05_structural_cases")


def main() -> None:
    style()
    overall_plot(pd.read_csv(TABLES / "mean_performance.csv"))
    learning_plot(pd.read_csv(TABLES / "mean_paired_effects.csv"))
    environment_plot(pd.read_csv(TABLES / "matched_environment_effects.csv"))
    gap_plot(pd.read_csv(TABLES / "split_gap.csv"))
    cases_plot(pd.read_csv(TABLES / "cases.csv"))
    print(f"Wrote five scientific figures to {FIGURES}")


if __name__ == "__main__":
    main()
