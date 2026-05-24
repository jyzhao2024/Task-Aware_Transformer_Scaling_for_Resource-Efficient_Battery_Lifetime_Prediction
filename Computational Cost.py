import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, NullFormatter


ROOT_DIR = r"./our data"
SAVE_DIR = ROOT_DIR
SAVE_NAME = "./result/figures"

N_TEST_FOR_TIME = 20         
USE_TOTAL_FLOPS_K = True      


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_hidden_dim(filename):

    m = re.search(r'Hidden(\d+)', filename)
    if m:
        return int(m.group(1))
    return None


def safe_r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = safe_r2_score(y_true, y_pred)

    denom = np.sum(np.abs(y_true))
    if denom == 0:
        wmape = np.nan
    else:
        wmape = np.sum(np.abs(y_true - y_pred)) / denom

    return {
        "RMSE": rmse,
        "R2": r2,
        "WMAPE": wmape,
        "MAE": mae
    }


def get_stat_value(stat_dict, candidate_keys, default=np.nan):

    for k in candidate_keys:
        if k in stat_dict:
            return stat_dict[k]
    return default


def extract_rul_metrics_from_res_dict(res_dict):

    y_true_all = []
    y_pred_all = []

    for key, value in res_dict.items():
        if str(key).startswith("_"):
            continue
        if not isinstance(value, dict):
            continue
        if "rul" not in value:
            continue

        rul_block = value["rul"]

        if "true" not in rul_block or "transfer" not in rul_block:
            continue

        y_true = np.asarray(rul_block["true"]).reshape(-1)
        y_pred = np.asarray(rul_block["transfer"]).reshape(-1)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) > 0:
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

    if len(y_true_all) == 0:
        raise ValueError("res_dict 中未找到有效的 rul.true 和 rul.transfer 数据。")

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    return compute_metrics(y_true_all, y_pred_all)


def collect_file_pairs(root_dir):
    all_files = os.listdir(root_dir)

    stat_files = {}
    res_files = {}

    for fn in all_files:
        if not fn.endswith(".pkl"):
            continue

        h = parse_hidden_dim(fn)
        if h is None:
            continue

        full_path = os.path.join(root_dir, fn)

        if fn.startswith("statistics_"):
            stat_files[h] = full_path
        elif fn.startswith("res_dict_"):
            res_files[h] = full_path

    hidden_dims = sorted(set(stat_files.keys()) & set(res_files.keys()), reverse=True)

    if len(hidden_dims) == 0:
        raise FileNotFoundError("没有找到可匹配的 statistics_*.pkl 和 res_dict_*.pkl 文件。")

    pairs = []
    for h in hidden_dims:
        pairs.append({
            "hidden_dim": h,
            "statistics_path": stat_files[h],
            "res_dict_path": res_files[h]
        })

    return pairs


def build_summary(root_dir):
    pairs = collect_file_pairs(root_dir)
    summary = []

    for item in pairs:
        h = item["hidden_dim"]

        stat_dict = load_pkl(item["statistics_path"])
        res_dict = load_pkl(item["res_dict_path"])

        metric_dict = extract_rul_metrics_from_res_dict(res_dict)

        # =========================
        # Model parameters
        # =========================
        model_params = get_stat_value(
            stat_dict,
            ["model_parameters", "Model parameters", "params", "total_parameters"],
            default=np.nan
        )


        inference_avg = get_stat_value(
            stat_dict,
            ["avg_inference_time_s", "Inference time (s)", "inference_time_per_sample_s"],
            default=np.nan
        )

        inference_total = get_stat_value(
            stat_dict,
            ["inference_time_s", "Total inference time (s)", "total_inference_time_s"],
            default=np.nan
        )

        if np.isfinite(inference_avg):
            inference_time = inference_avg
        elif np.isfinite(inference_total):
            if inference_total > 0.5:
                inference_time = inference_total / N_TEST_FOR_TIME
            else:
                inference_time = inference_total
        else:
            inference_time = np.nan


        all_flops_k = get_stat_value(
            stat_dict,
            ["all_test_samples_flops_K"],
            default=np.nan
        )

        all_flops_m = get_stat_value(
            stat_dict,
            ["all_test_samples_flops_M"],
            default=np.nan
        )

        all_flops_g = get_stat_value(
            stat_dict,
            ["all_test_samples_flops_G"],
            default=np.nan
        )

        flops_k = get_stat_value(
            stat_dict,
            ["FLOPs (K)", "flops_per_sample_K"],
            default=np.nan
        )

        flops_raw = get_stat_value(
            stat_dict,
            ["flops_per_sample"],
            default=np.nan
        )

        flops_m = get_stat_value(
            stat_dict,
            ["flops_per_sample_M"],
            default=np.nan
        )

        flops_g = get_stat_value(
            stat_dict,
            ["flops_per_sample_G"],
            default=np.nan
        )

        if USE_TOTAL_FLOPS_K and np.isfinite(all_flops_k):
            flops = all_flops_k
        elif USE_TOTAL_FLOPS_K and np.isfinite(all_flops_m):
            flops = all_flops_m * 1e3
        elif USE_TOTAL_FLOPS_K and np.isfinite(all_flops_g):
            flops = all_flops_g * 1e6
        elif np.isfinite(flops_k):
            flops = flops_k
        elif np.isfinite(flops_raw):
            flops = flops_raw / 1e3
        elif np.isfinite(flops_m):
            flops = flops_m * 1e3
        elif np.isfinite(flops_g):
            flops = flops_g * 1e6
        else:
            flops = np.nan

        summary.append({
            "H": h,
            "Params": float(model_params),
            "Params_M": float(model_params) / 1e6 if np.isfinite(model_params) else np.nan,
            "RMSE": float(metric_dict["RMSE"]),
            "R2": float(metric_dict["R2"]),
            "WMAPE": float(metric_dict["WMAPE"]),
            "MAE": float(metric_dict["MAE"]),
            "Inference": float(inference_time),
            "FLOPs": float(flops), 
        })

    summary = sorted(summary, key=lambda x: x["H"], reverse=True)
    return summary



def add_metric_note(ax, text, loc="upper right"):
   
    return


def add_point_labels(
    ax,
    x,
    y,
    Hs,
    params_m,
    values,
    metric_name,
    skip_h=None
):


    if skip_h is None:
        skip_h = []

    fmt_map = {
        "RMSE": "{:.0f}",
        "R2": "{:.3f}",
        "WMAPE": "{:.2f}",
        "MAE": "{:.1f}",
        "Inference": "{:.3f}",
        "FLOPs": "{:,.0f}"
    }

    offsets = {
        "RMSE": {
            4: (18, -44),
            8: (18, -44),
            16: (24, -44),
            32: (24, 34),
            64: (24, 34),
            128: (0, -30),
            256: (-36, -2),
        },
        "R2": {
            4: (24, 34),
            8: (24, 30),
            16: (24, -30),
            32: (28, -30),
            64: (28, -30),
            128: (0, -30),
            256: (-30, -24),
        },
        "WMAPE": {
            4: (28, -44),
            8: (24, 34),
            16: (26, 34),
            32: (26, 34),
            64: (26, 34),
            128: (0, -30),
            256: (-30, -2),
        },
        "MAE": {
            4: (24, -34),
            8: (26, 34),
            16: (26, 34),
            32: (26, 34),
            64: (26, 34),
            128: (0, -30),
            256: (-34, -2),
        },
        "Inference": {
            4: (26, 34),
            8: (26, -30),
            16: (28, 34),
            32: (38, 34),
            64: (28, 34),
            128: (28, -30),
            256: (-30, 16),
        },
        "FLOPs": {
            4: (28, -26),
            8: (28, 20),
            16: (28, 20),
            32: (50, 134),
            64: (30, 34),
            128: (50, -20),
            256: (-50, 4),
        }
    }

    value_fmt = fmt_map[metric_name]

    for xi, yi, h, pm, val in zip(x, y, Hs, params_m, values):
        h = int(h)

        if h in skip_h:
            continue

        dx, dy = offsets[metric_name].get(h, (20, 16))
        text = f"H={h}\n{pm:.4f}M\n{value_fmt.format(val)}"

        ax.annotate(
    text,
    xy=(xi, yi),
    xytext=(dx, dy),
    textcoords="offset points",
    ha="center",
    va="center",
    fontsize=14,   
    color="black",
    bbox=dict(
        boxstyle="round,pad=0.10",
        fc="white",
        ec="none",
        alpha=0.92
    ),
    zorder=20
)

def highlight_callout(ax, x, y, label, color, xytext):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext,
        textcoords="data",
        ha="center",
        va="center",
        fontsize=14,   
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec=color,
            linestyle="--",
            lw=1.2
        ),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.2,
            linestyle="--",
            shrinkA=4,
            shrinkB=6
        ),
        zorder=30
    )


def setup_common_axis(ax):
    ax.set_xscale("log")

    xticks = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00]
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        ["0.01", "0.02", "0.05", "0.10", "0.20", "0.50", "1.00", "2.00"],
        fontsize=18
    )

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        color="#bfbfbf",
        alpha=0.55,
        dashes=(3, 3)
    )

    ax.set_xlabel("Model parameters (M)", fontsize=18, labelpad=5)

    for spine in ax.spines.values():
        spine.set_color("#9a9a9a")
        spine.set_linewidth(0.9)

 
    ax.tick_params(axis="both", labelsize=18, direction="out", length=3)



def plot_simplified_figure(summary, save_path):
    Hs = np.array([d["H"] for d in summary])
    x = np.array([d["Params_M"] for d in summary], dtype=float)

    rmse = np.array([d["RMSE"] for d in summary], dtype=float)
    r2 = np.array([d["R2"] for d in summary], dtype=float)
    wmape = np.array([d["WMAPE"] for d in summary], dtype=float)
    mae = np.array([d["MAE"] for d in summary], dtype=float)
    inf_t = np.array([d["Inference"] for d in summary], dtype=float)
    flops = np.array([d["FLOPs"] for d in summary], dtype=float)

    base_color = "#145db2"
    best_acc_color = "#1abc9c"
    best_balance_color = "#ff7f0e"
    title_color = "black"

    idx_h128 = np.where(Hs == 128)[0][0] if 128 in Hs else None
    idx_h32 = np.where(Hs == 32)[0][0] if 32 in Hs else None
    idx_best_inf = int(np.nanargmin(inf_t))
    idx_best_flops = int(np.nanargmin(flops))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, axes = plt.subplots(3, 2, figsize=(16, 10.8))
    axes = axes.flatten()

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.95,
        bottom=0.08,
        hspace=0.34,
        wspace=0.16
    )

    def draw_scatter(ax, y, metric_name, title, ylabel, ylim, note_text="Lower is better"):
        ax.scatter(x, y, s=52, color=base_color, zorder=4)

        if idx_h32 is not None:
            ax.scatter(x[idx_h32], y[idx_h32], s=72, color=best_balance_color, zorder=5)

        if idx_h128 is not None:
            ax.scatter(x[idx_h128], y[idx_h128], s=72, color=best_acc_color, zorder=5)

        setup_common_axis(ax)

        ax.set_title(
            title,
            fontsize=22,
            color=title_color,
            fontweight="bold",
            loc="left",
            pad=8
        )

        ax.set_ylabel(ylabel, fontsize=20, labelpad=6)
        ax.set_ylim(*ylim)
        ax.set_xlim(0.008, 2.25)

        add_metric_note(ax, note_text)
        add_point_labels(ax, x, y, Hs, x, y, metric_name)

    # =====================================================
    # (a) RMSE
    # =====================================================
    ax = axes[0]
    draw_scatter(
        ax,
        rmse,
        "RMSE",
        "a",
        "RMSE",
        (80, 160),
        "Lower is better"
    )

    if idx_h32 is not None:
        highlight_callout(
            ax,
            x[idx_h32],
            rmse[idx_h32],
            "Best balance",
            best_balance_color,
            (0.1, 90)
        )

    if idx_h128 is not None:
        highlight_callout(
            ax,
            x[idx_h128],
            rmse[idx_h128],
            "Best accuracy",
            best_acc_color,
            (0.88, 140)
        )

    # =====================================================
    # (b) R2
    # =====================================================
    ax = axes[1]
    draw_scatter(
        ax,
        r2,
        "R2",
        "b",
        "R$^2$",
        (0.90, 1.00),
        "Higher is better"
    )

    if idx_h32 is not None:
        highlight_callout(
            ax,
            x[idx_h32],
            r2[idx_h32],
            "Best balance",
            best_balance_color,
            (0.1, 0.984)
        )

    if idx_h128 is not None:
        highlight_callout(
            ax,
            x[idx_h128],
            r2[idx_h128],
            "Best accuracy",
            best_acc_color,
            (0.66, 0.99)
        )

    # =====================================================
    # (c) WMAPE
    # =====================================================
    ax = axes[2]
    draw_scatter(
        ax,
        wmape,
        "WMAPE",
        "c",
        "WMAPE",
        (0.00, 0.20),
        "Lower is better"
    )

    if idx_h32 is not None:
        highlight_callout(
            ax,
            x[idx_h32],
            wmape[idx_h32],
            "Best balance",
            best_balance_color,
            (0.055, 0.045)
        )

    if idx_h128 is not None:
        highlight_callout(
            ax,
            x[idx_h128],
            wmape[idx_h128],
            "Best accuracy",
            best_acc_color,
            (0.86, 0.135)
        )

    # =====================================================
    # (d) MAE
    # =====================================================
    ax = axes[3]
    draw_scatter(
        ax,
        mae,
        "MAE",
        "d",
        "MAE",
        (40, 160),
        "Lower is better"
    )

    if idx_h32 is not None:
        highlight_callout(
            ax,
            x[idx_h32],
            mae[idx_h32],
            "Best balance",
            best_balance_color,
            (0.055, 64)
        )

    if idx_h128 is not None:
        highlight_callout(
            ax,
            x[idx_h128],
            mae[idx_h128],
            "Best accuracy",
            best_acc_color,
            (0.90, 110)
        )

    # =====================================================
    # (e) Inference time
    # =====================================================
    ax = axes[4]

    ax.scatter(x, inf_t, s=52, color=base_color, zorder=4)
    ax.scatter(x[idx_best_inf], inf_t[idx_best_inf], s=72, color=best_acc_color, zorder=5)

    setup_common_axis(ax)

    ax.set_title(
        "e",
        fontsize=22,
        color=title_color,
        fontweight="bold",
        loc="left",
        pad=8
    )

    ax.set_ylabel("Inference time (s)", fontsize=20, labelpad=6)
    ax.set_xlim(0.008, 2.25)

    inf_min = np.nanmin(inf_t)
    inf_max = np.nanmax(inf_t)
    inf_pad = max((inf_max - inf_min) * 0.55, 0.004)

    e_low = max(0.0, inf_min - inf_pad)
    e_high = inf_max + inf_pad

    ax.set_ylim(e_low, e_high)

    add_metric_note(ax, "Lower is better")
    add_point_labels(ax, x, inf_t, Hs, x, inf_t, "Inference")


    highlight_callout(
        ax,
        x[idx_best_inf],
        inf_t[idx_best_inf],
        "Fastest inference",
        best_acc_color,
        (0.10, e_low + 0.12 * (e_high - e_low))
    )

    # =====================================================
    # (f) FLOPs
    # =====================================================
    ax = axes[5]

    flops_raw_for_label = flops.copy()
    flops_plot = flops / 1e8

 
    ax.scatter(x, flops_plot, s=52, color=base_color, zorder=4)

    if idx_h32 is not None:
        ax.scatter(x[idx_h32], flops_plot[idx_h32], s=72, color=best_balance_color, zorder=5)

    if idx_h128 is not None:
        ax.scatter(x[idx_h128], flops_plot[idx_h128], s=72, color=best_acc_color, zorder=5)

    ax.scatter(x[idx_best_flops], flops_plot[idx_best_flops], s=72, color=best_acc_color, zorder=5)

    setup_common_axis(ax)

    ax.set_title(
        "f",
        fontsize=22,
        color=title_color,
        fontweight="bold",
        loc="left",
        pad=8
    )

    ax.set_ylabel(r"FLOPs (K, $\times 10^8$)", fontsize=20, labelpad=6)
    ax.set_xlim(0.008, 2.25)

    flops_plot_max = np.nanmax(flops_plot)
    ax.set_ylim(0, flops_plot_max * 1.22)

 
    add_point_labels(
        ax,
        x,
        flops_plot,
        Hs,
        x,
        flops_raw_for_label,
        "FLOPs",
        skip_h=[4, 8, 16]
    )

    if idx_h32 is not None:
        highlight_callout(
            ax,
            x[idx_h32],
            flops_plot[idx_h32],
            "Best balance",
            best_balance_color,
            (0.2, flops_plot_max * 0.7)
        )

    if idx_h128 is not None:
        highlight_callout(
            ax,
            x[idx_h128],
            flops_plot[idx_h128],
            "Best accuracy",
            best_acc_color,
            (1.2, flops_plot_max * 0.58)
        )

        inset_ax = inset_axes(
            ax,
            width="44%", 
            height="58%", 
            loc="upper left",
            bbox_to_anchor=(0.05, 0.02, 0.92, 0.92),
            bbox_transform=ax.transAxes,
            borderpad=0.8
        )

        small_h_list = [4, 8, 16]
        small_idx = [np.where(Hs == h)[0][0] for h in small_h_list if h in Hs]

        inset_x = x[small_idx]
        inset_y = flops_plot[small_idx]

 
        inset_ax.scatter(inset_x, inset_y, s=60, color=base_color, zorder=4)

   
        if idx_best_flops in small_idx:
            inset_ax.scatter(
                x[idx_best_flops],
                flops_plot[idx_best_flops],
                s=78,
                color=best_acc_color,
                zorder=5
            )

  
        inset_ax.set_xscale("log")
        inset_ax.set_xlim(0.008, 0.065)


        y_small_max = np.nanmax(inset_y)
        inset_ax.set_ylim(0, 0.60)

        inset_ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        inset_ax.set_yticklabels(["0.0", "0.2", "0.5", "0.6"], fontsize=8)

        inset_ax.grid(
            True,
            axis="y",
            linestyle="--",
            color="#bfbfbf",
            alpha=0.50,
            dashes=(3, 3)
        )


        inset_ax.xaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05]))
        inset_ax.xaxis.set_major_formatter(FixedFormatter(["0.01", "0.02", "0.05"]))
        inset_ax.xaxis.set_minor_locator(NullLocator())
        inset_ax.xaxis.set_minor_formatter(NullFormatter())
        inset_ax.xaxis.offsetText.set_visible(False)

        inset_ax.tick_params(axis="x", labelsize=8, direction="out", length=2, pad=2)
        inset_ax.tick_params(axis="y", labelsize=8, direction="out", length=2)

        for spine in inset_ax.spines.values():
            spine.set_color("#9a9a9a")
            spine.set_linewidth(0.8)

        inset_ax.set_title(
            "Zoomed view",
            fontsize=12,
            color="black",
            pad=3
        )


        inset_label_cfg = {
            4: {"offset": (18, 8), "ha": "left", "va": "center"},
            8: {"offset": (4, 12), "ha": "center", "va": "bottom"},
            16: {"offset": (-10, 8), "ha": "left", "va": "bottom"},
        }

        for idx in small_idx:
            h = int(Hs[idx])
            cfg = inset_label_cfg.get(h, {"offset": (12, 10), "ha": "left", "va": "bottom"})
            dx, dy = cfg["offset"]

            inset_ax.annotate(
                f"H={h}\n{x[idx]:.4f}M\n{flops_raw_for_label[idx]:,.0f}",
                xy=(x[idx], flops_plot[idx]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=cfg["ha"],
                va=cfg["va"],
                fontsize=10,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    fc="white",
                    ec="none",
                    alpha=0.92
                ),
                zorder=15
            )

  
        if idx_best_flops in small_idx:
            inset_ax.annotate(
                "Lowest FLOPs",
                xy=(x[idx_best_flops], flops_plot[idx_best_flops]),
                xytext=(0.012, y_small_max * 1.15),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=10,
                color=best_acc_color,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="white",
                    ec=best_acc_color,
                    linestyle="--",
                    lw=1.0
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color=best_acc_color,
                    lw=1.0,
                    linestyle="--",
                    shrinkA=4,
                    shrinkB=5
                ),
                zorder=20
            )

     
        try:
            mark_inset(
                ax,
                inset_ax,
                loc1=3,
                loc2=4,
                fc="none",
                ec="#9a9a9a",
                lw=0.8,
                linestyle="--"
            )
        except Exception:
            pass

    plt.savefig(save_path.replace(".png", ".tif"), dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    summary = build_summary(ROOT_DIR)

    print("========== Summary ==========")
    for item in summary:
        print(item)

    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    plot_simplified_figure(summary, save_path)

   
    print("TIF:", save_path.replace(".png", ".tif"))
 
