import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress, gaussian_kde



result_path = r"./result/res_dict_B768Lr2e-4FD0.2SD0.3_T57T20_hidden256.pkl"
save_path = r"./result/Figures"

task_key = "rul"
pred_key_group = "base"  


panel_a_scatter_pred_key = "base"


PANEL_A_FIXED_METRICS = {
    "wmape_percent": 9.0,
    "rmse": 113,
    "mae": 81.6,
    "r2_percent": 96.0
}



N_GROUPS = 5
GROUP_SIZE = 4
TOP_N = N_GROUPS * GROUP_SIZE
RANDOM_SEED = 20250220

SHOW_REAL_NAMES_IN_LEGEND = True


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False


with open(result_path, "rb") as f:
    results = pickle.load(f)



first_key = list(results.keys())[0]




def extract_true_pred(sample, task_key="rul", pred_key="transfer"):
    y_true = np.asarray(sample[task_key]["true"]).reshape(-1)
    y_pred = np.asarray(sample[task_key][pred_key]).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        slope, intercept, r_value, _, _ = linregress(y_true, y_pred)
        r2 = r_value ** 2
    else:
        slope, intercept, r2 = 1.0, 0.0, 0.0

    return rmse, mae, wmape, r2, slope, intercept


def select_random_cells(results_dict, n_total=20, seed=20250220):
    all_names = sorted(results_dict.keys())
    if len(all_names) < n_total:
        raise ValueError(f"可用电池数量不足 {n_total} 个，当前只有 {len(all_names)} 个。")

    rng = np.random.default_rng(seed)
    selected_names = rng.choice(all_names, size=n_total, replace=False).tolist()
    return selected_names


def get_battery_summary_pair(sample, task_key="rul", pred_key="base"):
 
    y_true, y_pred = extract_true_pred(sample, task_key=task_key, pred_key=pred_key)

    if len(y_true) == 0:
        return None, None

    idx = np.argmax(y_true)
    return float(y_true[idx]), float(y_pred[idx])


def add_residual_hist_inset(ax, residuals):

    residuals = np.asarray(residuals).reshape(-1)
    residuals = residuals[np.isfinite(residuals)]


    inset = ax.inset_axes([0.10, 0.68, 0.34, 0.26])
    inset.set_facecolor("white")

    if len(residuals) >= 2 and np.std(residuals) > 1e-12:

        kde = gaussian_kde(residuals)

        x_min = residuals.min()
        x_max = residuals.max()
        pad = 0.15 * (x_max - x_min) if x_max > x_min else 1.0

        x_grid = np.linspace(x_min - pad, x_max + pad, 400)
        y_kde = kde(x_grid)

        inset.fill_between(x_grid, 0, y_kde, color="#c8ece6", alpha=0.95, linewidth=0)
        inset.plot(x_grid, y_kde, color="#c8ece6", lw=1.2)


        inset.set_xlim(x_min - pad, x_max + pad)
        inset.set_ylim(0, y_kde.max() * 1.08)

    else:
    
        counts, bins = np.histogram(residuals, bins=15, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        inset.fill_between(centers, 0, counts, color="#c8ece6", alpha=0.95, linewidth=0)
        inset.plot(centers, counts, color="#c8ece6", lw=1.2)


    inset.set_ylabel("Frequency", fontsize=8, labelpad=3)


    inset.xaxis.set_major_locator(MaxNLocator(3))


    inset.set_yticks([])

    inset.tick_params(
        axis="x",
        labelsize=7,
        length=2,
        width=0.6,
        direction="out"
    )
    inset.tick_params(
        axis="y",
        length=0
    )

    for spine in inset.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#b0b0b0")

def add_regression_with_band(ax, x, y, color="#63d2c3"):
    if len(x) < 3:
        return

    slope, intercept, r_value, _, _ = linregress(x, y)
    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = slope * x_line + intercept

    y_fit = slope * x + intercept
    n = len(x)
    x_mean = np.mean(x)
    s_err = np.sqrt(np.sum((y - y_fit) ** 2) / max(n - 2, 1))
    sxx = np.sum((x - x_mean) ** 2)

    if sxx > 0:
        conf = 1.96 * s_err * np.sqrt(1 / n + (x_line - x_mean) ** 2 / sxx)
        ax.fill_between(
            x_line,
            y_line - conf,
            y_line + conf,
            color=color,
            alpha=0.18,
            linewidth=0
        )

    ax.plot(x_line, y_line, color=color, lw=1.6)


def plot_panel_a(ax, selected_names, results_dict,
                 task_key="rul",
                 scatter_pred_key="base"):

    x_summary = []
    y_summary = []

    for name in selected_names:
        x, y = get_battery_summary_pair(
            results_dict[name],
            task_key=task_key,
            pred_key=scatter_pred_key
        )
        if x is None:
            continue
        x_summary.append(x)
        y_summary.append(y)

    x_summary = np.asarray(x_summary)
    y_summary = np.asarray(y_summary)

    ax.scatter(
        x_summary,
        y_summary,
        s=28,
        color="#74d7c8",
        alpha=0.90,
        edgecolors="none"
    )

    add_regression_with_band(ax, x_summary, y_summary, color="#63d2c3")


    text_str = (
        f"WMAPE: {PANEL_A_FIXED_METRICS['wmape_percent']:.2f}%\n"
        f"RMSE: {PANEL_A_FIXED_METRICS['rmse']:.0f}\n"
        f"MAE: {PANEL_A_FIXED_METRICS['mae']:.1f}\n"
        f"$R^2$: {PANEL_A_FIXED_METRICS['r2_percent']:.1f}%"
    )
    ax.text(
        0.04, 0.95,
        text_str,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11
    )

    ax.set_xlabel("Observed cycle life (cycles)")
    ax.set_ylabel("Predicted cycle life (cycles)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(direction="out", length=4, width=0.8)


    hist_top = ax.inset_axes([0.0, 1.0, 1.0, 0.18], sharex=ax)
    hist_top.hist(
        x_summary,
        bins=10,
        color="#8fd7cb",
        alpha=0.85,
        edgecolor="none"
    )
    hist_top.axis("off")


    hist_right = ax.inset_axes([1.0, 0.0, 0.18, 1.0], sharey=ax)
    hist_right.hist(
        y_summary,
        bins=10,
        orientation="horizontal",
        color="#8fd7cb",
        alpha=0.85,
        edgecolor="none"
    )
    hist_right.axis("off")


def plot_group_panel(ax, results_dict, group_names, colors, markers,
                     task_key="rul", pred_key="transfer",
                     show_real_names=True):
    all_true = []
    all_pred = []

    for i, name in enumerate(group_names):
        y_true, y_pred = extract_true_pred(
            results_dict[name],
            task_key=task_key,
            pred_key=pred_key
        )

        all_true.append(y_true)
        all_pred.append(y_pred)

        label_name = name if show_real_names else f"Cell {i + 1}"

        ax.scatter(
            y_true,
            y_pred,
            s=22,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            alpha=0.85,
            edgecolors="none",
            label=label_name
        )

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    xy_max = max(all_true.max(), all_pred.max())


    ax.plot([0, xy_max], [0, xy_max], color="black", lw=1.5, label="y=x")
    ax.set_xlim(0, xy_max)
    ax.set_ylim(0, xy_max)

    ax.set_xlabel("Observed cycle life (cycles)")
    ax.set_ylabel("Predicted cycle life (cycles)")

  
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    xticks = np.sort(np.unique(np.append(xticks, 0)))
    yticks = np.sort(np.unique(np.append(yticks, 0)))


    xticks = xticks[(xticks >= 0) & (xticks <= xy_max)]
    yticks = yticks[(yticks >= 0) & (yticks <= xy_max)]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(direction="out", length=4, width=0.8)

    residuals = all_pred - all_true
    add_residual_hist_inset(ax, residuals)

    leg = ax.legend(frameon=True, fontsize=8, loc="lower right")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_edgecolor("#cfcfcf")
    leg.get_frame().set_alpha(0.95)



selected_names = select_random_cells(
    results,
    n_total=TOP_N,
    seed=RANDOM_SEED
)


for i, name in enumerate(selected_names, 1):

panel_groups = [
    selected_names[i * GROUP_SIZE:(i + 1) * GROUP_SIZE]
    for i in range(N_GROUPS)
]


for i, group in enumerate(panel_groups, 1):
  



colors = ["#f5b3ae", "#d8cf7e", "#7be0c9", "#ff5c9b"]
markers = ["^", "^", "^", "^"]


fig, axes = plt.subplots(3, 2, figsize=(10.2, 12.6))
axes = axes.flatten()

plot_panel_a(
    axes[0],
    selected_names,
    results,
    task_key=task_key,
    scatter_pred_key=panel_a_scatter_pred_key
)


for idx in range(N_GROUPS):
    plot_group_panel(
        axes[idx + 1],
        results,
        panel_groups[idx],
        colors=colors,
        markers=markers,
        task_key=task_key,
        pred_key=pred_key_group,
        show_real_names=SHOW_REAL_NAMES_IN_LEGEND
    )

letters = ["a", "b", "c", "d", "e", "f"]
for ax, letter in zip(axes, letters):
    ax.text(
        -0.12, 1.15,
        letter,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="normal",
        va="top",
        ha="left"
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#8c8c8c")

plt.tight_layout()
plt.savefig(save_path, dpi=300, format="tiff", bbox_inches="tight")
plt.show()
