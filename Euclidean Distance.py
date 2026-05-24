import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle


FILE_MAP = {
    "37": r"./result/res_dict_B768Lr2e-4FD0.2SD0.3_T40T37_datasize.pkl",
    "47": r"./result/res_dict_B768Lr5e-4FD0.2SD0.3_T30T47_datasize.pkl",
    "57": r"./result/res_dict_B768Lr5e-4FD0.2SD0.3_T20T57_datasize.pkl",
}

SAVE_PATH = r"./result/figures"

TASK_KEY = "rul"       
PRED_KEY = "transfer"   


PRINT_SUMMARY = True

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.unicode_minus"] = False


COLORS = ["#79d3f5", "#f5a6a6", "#8894ff"]  

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def calc_metrics(y_true, y_pred):

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan, np.nan

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    denom = np.sum(np.abs(y_true))
    wmape = np.sum(np.abs(y_true - y_pred)) / denom if denom != 0 else np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return rmse, r2, wmape, mae


def extract_true_pred(sample, task_key="rul", pred_key="transfer"):

    if isinstance(sample, dict) and task_key in sample:
        task_part = sample[task_key]
        if isinstance(task_part, dict) and "true" in task_part and pred_key in task_part:
            y_true = np.asarray(task_part["true"]).reshape(-1)
            y_pred = np.asarray(task_part[pred_key]).reshape(-1)
            return y_true, y_pred

    if isinstance(sample, dict) and "true" in sample and pred_key in sample:
        y_true = np.asarray(sample["true"]).reshape(-1)
        y_pred = np.asarray(sample[pred_key]).reshape(-1)
        return y_true, y_pred

    raise KeyError(f"无法从 sample 中提取 {task_key}/{pred_key} 对应的 true/pred 数据。")


def collect_metrics_from_one_file(pkl_path, task_key="rul", pred_key="transfer"):
 
    results = load_pkl(pkl_path)

    rmse_list = []
    r2_list = []
    wmape_list = []
    mae_list = []

    if not isinstance(results, dict):
        raise TypeError(f"{pkl_path} 读取后不是 dict，请检查文件结构。")

    for name, sample in results.items():
        try:
            y_true, y_pred = extract_true_pred(sample, task_key=task_key, pred_key=pred_key)
            rmse, r2, wmape, mae = calc_metrics(y_true, y_pred)

            if np.isfinite(rmse):
                rmse_list.append(rmse)
            if np.isfinite(r2):
                r2_list.append(r2)
            if np.isfinite(wmape):
                wmape_list.append(wmape)
            if np.isfinite(mae):
                mae_list.append(mae)

        except Exception as e:
            print(f"[Warning] {os.path.basename(pkl_path)} -> {name} 处理失败: {e}")

    return {
        "RMSE": np.array(rmse_list),
        "R2": np.array(r2_list),
        "WMAPE": np.array(wmape_list),
        "MAE": np.array(mae_list),
    }



dataset_labels = list(FILE_MAP.keys())
all_metrics = {label: None for label in dataset_labels}

for label, path in FILE_MAP.items():
    metrics = collect_metrics_from_one_file(path, task_key=TASK_KEY, pred_key=PRED_KEY)
    all_metrics[label] = metrics

    if PRINT_SUMMARY:
        print("=" * 60)
        print(label)
        print(path)
        for k in ["RMSE", "R2", "WMAPE", "MAE"]:
            arr = metrics[k]
            print(f"{k:6s}: n={len(arr):2d}, mean={np.mean(arr):.4f}, std={np.std(arr):.4f}")


def draw_half_violin(ax, values, y0, color, xlim, violin_height=0.42, bw=0.25):

    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return

    xmin, xmax = xlim
    x_grid = np.linspace(xmin, xmax, 400)


    if np.std(values) < 1e-12:
        return

    kde = gaussian_kde(values, bw_method=bw)
    density = kde(x_grid)


    if density.max() > 0:
        density = density / density.max() * violin_height

    ax.fill_between(x_grid, y0, y0 + density, color=color, alpha=0.35, linewidth=0)
    ax.plot(x_grid, y0 + density, color=color, lw=1.3)
    ax.plot([xmin, xmax], [y0, y0], color=color, lw=1.0, alpha=0.8)


def draw_rug_scatter(ax, values, y0, color, scatter_offset=0.18, jitter_height=0.10):
 
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    rng = np.random.default_rng(2025)
    y = y0 - scatter_offset - rng.uniform(0.0, jitter_height, size=len(values))

    ax.scatter(
        values, y,
        s=10,
        color=color,
        alpha=0.95,
        edgecolors="none",
        zorder=3
    )


def draw_summary_box(ax, values, y0, color, box_height=0.08):

    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    q1 = np.percentile(values, 25)
    median = np.percentile(values, 50)
    q3 = np.percentile(values, 75)
    mean = np.mean(values)

    rect = Rectangle(
        (q1, y0 - box_height / 2),
        q3 - q1,
        box_height,
        facecolor=color,
        edgecolor=color,
        alpha=0.20,
        linewidth=1.0,
        zorder=4
    )
    ax.add_patch(rect)

    ax.plot([median, median], [y0 - box_height / 2, y0 + box_height / 2],
            color=color, lw=1.3, zorder=5)

    ax.scatter([mean], [y0], marker="s", s=22, color=color, zorder=6)


def draw_one_panel(ax, metric_dict, metric_name, xlabel, letter, colors,
                   xlim=None, xticks=None, bw=0.25):

 
    labels_order = ["57", "47", "37"]
    y_positions = [3, 2, 1]

 
    if xlim is None:
        all_vals = np.concatenate([metric_dict[k] for k in labels_order])
        xmin = min(0, np.min(all_vals) * 0.95)
        xmax = np.max(all_vals) * 1.15
        xlim = (xmin, xmax)

    for i, (label, y0) in enumerate(zip(labels_order, y_positions)):
        values = metric_dict[label]
        c = colors[2 - i] 

        draw_half_violin(ax, values, y0, c, xlim=xlim, violin_height=0.42, bw=bw)
        draw_rug_scatter(ax, values, y0, c, scatter_offset=0.16, jitter_height=0.10)
        draw_summary_box(ax, values, y0, c, box_height=0.08)

    ax.set_xlim(xlim)
    ax.set_ylim(0.4, 3.6)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["37", "47", "57"], rotation=0, ha="right", va="center")
    ax.set_xlabel(xlabel)

    if xticks is not None:
        ax.set_xticks(xticks)


    ax.text(
        -0.12, 1.02, letter,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="normal",
        va="bottom",
        ha="left"
    )

 
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#b0b0b0")

    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.grid(False)



rmse_dict = {label: all_metrics[label]["RMSE"] for label in dataset_labels}
r2_dict = {label: all_metrics[label]["R2"] for label in dataset_labels}
wmape_dict = {label: all_metrics[label]["WMAPE"] for label in dataset_labels}
mae_dict = {label: all_metrics[label]["MAE"] for label in dataset_labels}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

draw_one_panel(
    axes[0], rmse_dict, "RMSE", "RMSE", "a", COLORS,
    xlim=(0, 600), xticks=np.arange(0, 601, 100), bw=0.25
)

draw_one_panel(
    axes[1], r2_dict, r"$R^2$", r"$R^2$", "b", COLORS,
    xlim=(0.3, 1.2), xticks=np.arange(0.3, 1.21, 0.1), bw=0.25
)

draw_one_panel(
    axes[2], wmape_dict, "WMAPE", "WMAPE", "c", COLORS,
    xlim=(0.0, 0.45), xticks=np.arange(0.0, 0.46, 0.1), bw=0.25
)

draw_one_panel(
    axes[3], mae_dict, "MAE", "MAE", "d", COLORS,
    xlim=(0, 500), xticks=np.arange(0, 501, 100), bw=0.25
)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, format="tiff", bbox_inches="tight")
plt.show()
