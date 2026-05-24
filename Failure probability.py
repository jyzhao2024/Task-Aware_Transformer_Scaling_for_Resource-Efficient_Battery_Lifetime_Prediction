import os
import math
import pickle
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DATA_DIR = r"./our data/Failure probability"

SAVE_DIR_FIG = r"./result/Fiures"

os.makedirs(SAVE_DIR_FIG5, exist_ok=True)
os.makedirs(SAVE_DIR_FIGS7, exist_ok=True)


RES_DICT_PATH = os.path.join(DATA_DIR, "res_dict_B768_Lr2e-4FD0.2SD0.3_T57T20_failure.pkl")
TEST_LOADER_PATH = r".data\LFP\test_loader_weibull.pkl"

FIG_BATTERIES = [
    "4-1", "3-5",
    "9-8", "5-3",
    "5-2", "5-1",
    "7-2", "1-3"
]


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.dpi"] = 150


COLOR_REF = "#5FC4D8"     
COLOR_EST = "#8ED9EA"     
COLOR_CI = "#BFE9E1"    
GRID_COLOR = "#D9D9D9"



def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def moving_average(y, win=1):
  
    y = np.asarray(y, dtype=float)
    if win <= 1 or len(y) < win:
        return y
    kernel = np.ones(win) / win
    y_pad = np.pad(y, (win // 2, win - 1 - win // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def clean_and_sort_curve(x, y_ref, y_est, ci_low, ci_high):
   
    x = np.asarray(x, dtype=float).reshape(-1)
    y_ref = np.asarray(y_ref, dtype=float).reshape(-1)
    y_est = np.asarray(y_est, dtype=float).reshape(-1)
    ci_low = np.asarray(ci_low, dtype=float).reshape(-1)
    ci_high = np.asarray(ci_high, dtype=float).reshape(-1)

    min_len = min(len(x), len(y_ref), len(y_est), len(ci_low), len(ci_high))
    x = x[:min_len]
    y_ref = y_ref[:min_len]
    y_est = y_est[:min_len]
    ci_low = ci_low[:min_len]
    ci_high = ci_high[:min_len]

    mask = (
        np.isfinite(x) &
        np.isfinite(y_ref) &
        np.isfinite(y_est) &
        np.isfinite(ci_low) &
        np.isfinite(ci_high)
    )

    x = x[mask]
    y_ref = y_ref[mask]
    y_est = y_est[mask]
    ci_low = ci_low[mask]
    ci_high = ci_high[mask]


    y_ref = np.clip(y_ref, 0.0, 1.0)
    y_est = np.clip(y_est, 0.0, 1.0)
    ci_low = np.clip(ci_low, 0.0, 1.0)
    ci_high = np.clip(ci_high, 0.0, 1.0)


    tmp_low = np.minimum(ci_low, ci_high)
    tmp_high = np.maximum(ci_low, ci_high)
    ci_low, ci_high = tmp_low, tmp_high


    order = np.argsort(x)
    x = x[order]
    y_ref = y_ref[order]
    y_est = y_est[order]
    ci_low = ci_low[order]
    ci_high = ci_high[order]

    return x, y_ref, y_est, ci_low, ci_high


def extract_battery_curve(res_dict, test_loader_dict, battery_name, prob_col=-1, smooth_win=1):
   
    if battery_name not in res_dict:
        raise KeyError(f"{battery_name} 不在 res_dict 中。")

    if battery_name not in test_loader_dict:
        raise KeyError(f"{battery_name} 不在 test_loader_weibull 中。")


    x = res_dict[battery_name]["rul"]["true"]
  
    y_ref = res_dict[battery_name]["soh"]["true"]

    y_est = res_dict[battery_name]["soh"]["transfer"]

    ci_low_all = test_loader_dict[battery_name]["ci_low"]
    ci_high_all = test_loader_dict[battery_name]["ci_high"]

    if ci_low_all.ndim == 2:
        ci_low = ci_low_all[:, prob_col]
        ci_high = ci_high_all[:, prob_col]
    else:
        ci_low = ci_low_all
        ci_high = ci_high_all

    x, y_ref, y_est, ci_low, ci_high = clean_and_sort_curve(
        x, y_ref, y_est, ci_low, ci_high
    )

    y_ref = moving_average(y_ref, smooth_win)
    y_est = moving_average(y_est, smooth_win)
    ci_low = moving_average(ci_low, smooth_win)
    ci_high = moving_average(ci_high, smooth_win)

    y_ref = np.clip(y_ref, 0, 1)
    y_est = np.clip(y_est, 0, 1)
    ci_low = np.clip(ci_low, 0, 1)
    ci_high = np.clip(ci_high, 0, 1)

    return x, y_ref, y_est, ci_low, ci_high

def nice_upper_bound(xmax):
    
    if xmax <= 0:
        return 1
    magnitude = 10 ** int(np.floor(np.log10(xmax)))
    upper = math.ceil(xmax / magnitude) * magnitude
    if upper < xmax * 1.02:
        upper += magnitude
    return upper

def plot_single_panel(ax, battery_name, x, y_ref, y_est, ci_low, ci_high, panel_label):
  
    ax.fill_between(
        x, ci_low, ci_high,
        color=COLOR_CI,
        alpha=0.65,
        linewidth=0.0,
        label="95% CI",
        zorder=1
    )

    ax.plot(
        x, y_ref,
        color=COLOR_REF,
        linewidth=1.6,
        label="Weibull-based failure probability",
        zorder=3
    )
    
    ax.plot(
        x, y_est,
        color=COLOR_EST,
        linewidth=1.6,
        label="Estimated failure probability",
        zorder=4
    )

    xmax = np.max(x) if len(x) > 0 else 1.0
    ymax = max(
        np.max(y_ref) if len(y_ref) > 0 else 0,
        np.max(y_est) if len(y_est) > 0 else 0,
        np.max(ci_high) if len(ci_high) > 0 else 0
    )
    ymax = min(1.05, max(0.12, ymax * 1.05))

    ax.set_xlim(0, nice_upper_bound(xmax))
    ax.set_ylim(0, ymax)
    
    ax.set_title(f"Battery {battery_name} + 95% CI", fontsize=9, pad=4)
    ax.set_xlabel("Remaining useful life", fontsize=9, labelpad=2)
    ax.set_ylabel("Failure probability", fontsize=9, labelpad=2)


    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.8)


    ax.tick_params(axis="both", labelsize=8, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    handles, labels = ax.get_legend_handles_labels()

    label_to_handle = dict(zip(labels, handles))
    ordered_labels = [
        "Weibull-based failure probability",
        "Estimated failure probability",
        "95% CI"
    ]
    ordered_handles = [label_to_handle[l] for l in ordered_labels if l in label_to_handle]

    ax.legend(
        ordered_handles,
        ordered_labels,
        fontsize=6.3,
        loc="upper right",
        frameon=False,
        handlelength=1.8,
        borderpad=0.2
    )
    ax.text(
        -0.10, 1.03, panel_label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="normal",
        va="bottom",
        ha="left"
    )

def make_multi_panel_figure(
    res_dict,
    test_loader_dict,
    battery_list,
    nrows,
    ncols,
    save_dir,
    fig_basename,
    fig_size=(7.2, 9.6),
    prob_col=-1,
    smooth_win=1
):
   
    letters = list(string.ascii_lowercase)

    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)
    axes = np.array(axes).reshape(-1)

    for i, battery_name in enumerate(battery_list):
        ax = axes[i]

        try:
            x, y_ref, y_est, ci_low, ci_high = extract_battery_curve(
                res_dict=res_dict,
                test_loader_dict=test_loader_dict,
                battery_name=battery_name,
                prob_col=prob_col,
                smooth_win=smooth_win
            )

            plot_single_panel(
                ax=ax,
                battery_name=battery_name,
                x=x,
                y_ref=y_ref,
                y_est=y_est,
                ci_low=ci_low,
                ci_high=ci_high,
                panel_label=letters[i]
            )

        except Exception as e:
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                f"{letters[i]}) Battery {battery_name}\nLoad failed:\n{e}",
                ha="center", va="center", fontsize=9
            )

        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.32, hspace=0.35)

    tif_path = os.path.join(save_dir, fig_basename + ".tif")

    fig.savefig(tif_path, dpi=300, bbox_inches="tight", facecolor="white")

    plt.show()
    plt.close(fig)

def main():

    if not os.path.exists(RES_DICT_PATH):
        raise FileNotFoundError(f"未找到文件：{RES_DICT_PATH}")

    if not os.path.exists(TEST_LOADER_PATH):
        raise FileNotFoundError(f"未找到文件：{TEST_LOADER_PATH}")

    res_dict = load_pkl(RES_DICT_PATH)
    test_loader_dict = load_pkl(TEST_LOADER_PATH)


    available_batteries = sorted(list(res_dict.keys()))
    print(available_batteries)
    print()

    make_multi_panel_figure(
        res_dict=res_dict,
        test_loader_dict=test_loader_dict,
        battery_list=FIG5_BATTERIES,
        nrows=4,
        ncols=2,
        save_dir=SAVE_DIR_FIG5,
        fig_basename="Fig5_failure_probability_8_batteries",
        fig_size=(7.3, 10.4),
        prob_col=-1,    
        smooth_win=1    
    )

if __name__ == "__main__":
    main()


