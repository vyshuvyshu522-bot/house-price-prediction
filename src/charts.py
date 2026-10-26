"""Dashboard charts, rendered once at training time as PNG files."""
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ACCENT, ACCENT_2, GRID = "#2563eb", "#f97316", "#e2e8f0"


def _money_axis(axis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1e6:.1f}M" if v >= 1e6 else f"${v / 1e3:.0f}K"))


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_actual_vs_predicted(y_true, y_pred, model_name, path):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.scatter(y_true, y_pred, s=7, alpha=0.35, color=ACCENT, edgecolors="none")
    limit = max(y_true.max(), y_pred.max())
    ax.plot([0, limit], [0, limit], color=ACCENT_2, linewidth=1.5, label="Perfect prediction")
    ax.set(xlabel="Actual price", ylabel="Predicted price", title=f"Actual vs predicted (test set) - {model_name}")
    _money_axis(ax.xaxis), _money_axis(ax.yaxis)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_price_distribution(prices, path):
    prices = np.asarray(prices)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.hist(prices, bins=60, color=ACCENT, alpha=0.9)
    ax.axvline(np.median(prices), color=ACCENT_2, linewidth=1.5, label=f"Median ${np.median(prices):,.0f}")
    ax.set(xlabel="Sale price", ylabel="Number of houses", title="House price distribution (cleaned dataset)")
    _money_axis(ax.xaxis)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_feature_importance(importances: dict, model_name, path):
    items = sorted(importances.items(), key=lambda kv: kv[1])
    names, values = [k for k, _ in items], [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.barh(names, values, color=ACCENT)
    ax.set(xlabel="Drop in R² when the feature is shuffled",
           title=f"Permutation feature importance (test set) - {model_name}")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
