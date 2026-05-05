# import os
# import re
# import glob
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from matplotlib.ticker import FormatStrFormatter
#
#
# # ============================================================
# # 全局字体设置
# # ============================================================
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 24
# plt.rcParams["axes.unicode_minus"] = False
#
#
# # ============================================================
# # 路径设置
# # ============================================================
# RESULT_DIR = r"E:\蔡密生\CMS\赵老师\工作\工作15-学姐论文修改交接\FAST_2026\CODE\SOC input range"
# SAVE_DIR = r"E:\蔡密生\CMS\赵老师\工作\工作15-学姐论文修改交接\FAST_2026\CODE\论文图\SOC_metric_raincloud"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
#
# # ============================================================
# # 文件匹配
# # ============================================================
# FILE_PATTERN = "res_dict_*_SOC_0_*.pkl"
#
#
# # ============================================================
# # 结果结构设置
# # ============================================================
# RUL_KEY = "rul"
# TRUE_KEY = "true"
# PRED_KEY = "base"
#
#
# # ============================================================
# # 基本函数
# # ============================================================
# def load_obj_by_path(file_path):
#     with open(file_path, "rb") as f:
#         return pickle.load(f)
#
#
# def parse_soc_from_filename(file_path):
#     filename = os.path.basename(file_path)
#     match = re.search(r"SOC_0_(\d+)", filename)
#
#     if match is None:
#         return None
#
#     return int(match.group(1))
#
#
# def cal_wmape(predict, groundtruth):
#     predict = np.asarray(predict, dtype=float).reshape(-1)
#     groundtruth = np.asarray(groundtruth, dtype=float).reshape(-1)
#
#     n = min(len(predict), len(groundtruth))
#     predict = predict[:n]
#     groundtruth = groundtruth[:n]
#
#     mask = np.isfinite(predict) & np.isfinite(groundtruth)
#     predict = predict[mask]
#     groundtruth = groundtruth[mask]
#
#     if len(groundtruth) == 0:
#         return np.nan
#
#     total_value = np.sum(np.abs(groundtruth))
#
#     if total_value < 1e-12:
#         return np.nan
#
#     abs_error = np.abs(predict - groundtruth)
#
#     return np.sum(abs_error) / total_value
#
#
# def compute_single_cell_metrics(rul_true, rul_pred):
#     rul_true = np.asarray(rul_true, dtype=float).reshape(-1)
#     rul_pred = np.asarray(rul_pred, dtype=float).reshape(-1)
#
#     n = min(len(rul_true), len(rul_pred))
#
#     if n < 2:
#         return None
#
#     rul_true = rul_true[:n]
#     rul_pred = rul_pred[:n]
#
#     mask = np.isfinite(rul_true) & np.isfinite(rul_pred)
#     rul_true = rul_true[mask]
#     rul_pred = rul_pred[mask]
#
#     if len(rul_true) < 2:
#         return None
#
#     rmse = float(np.sqrt(mean_squared_error(rul_true, rul_pred)))
#     r2 = float(r2_score(rul_true, rul_pred))
#     wmape = float(cal_wmape(rul_pred, rul_true))
#     mae = float(mean_absolute_error(rul_true, rul_pred))
#
#     return {
#         "RMSE": rmse,
#         "R2": r2,
#         "WMAPE": wmape,
#         "MAE": mae,
#         "n_points": len(rul_true)
#     }
#
#
# def sort_cell_keys(cell_keys):
#     """
#     尽量按照电池编号排序。
#     例如 7-5, 7-6, 8-1, 9-8。
#     """
#     def key_func(x):
#         nums = re.findall(r"\d+", str(x))
#
#         if len(nums) >= 2:
#             return int(nums[0]), int(nums[1])
#
#         if len(nums) == 1:
#             return int(nums[0]), 0
#
#         return 9999, 9999
#
#     return sorted(cell_keys, key=key_func)
#
#
# # ============================================================
# # 读取所有 SOC 文件并计算指标
# # ============================================================
# def collect_soc_metric_data(result_dir, file_pattern):
#     file_list = sorted(glob.glob(os.path.join(result_dir, file_pattern)))
#
#     if len(file_list) == 0:
#         raise FileNotFoundError(f"没有找到文件: {os.path.join(result_dir, file_pattern)}")
#
#     data = {
#         "RMSE": {},
#         "R2": {},
#         "WMAPE": {},
#         "MAE": {}
#     }
#
#     detail_rows = []
#     summary_rows = []
#
#     for file_path in file_list:
#         soc_end = parse_soc_from_filename(file_path)
#
#         if soc_end is None:
#             print(f"[Warning] 无法从文件名解析 SOC 范围，跳过: {file_path}")
#             continue
#
#         result = load_obj_by_path(file_path)
#
#         print("=" * 100)
#         print(f"File: {os.path.basename(file_path)}")
#         print(f"SOC range: {soc_end}%")
#
#         if not isinstance(result, dict):
#             print("[Warning] 当前文件内容不是 dict，跳过。")
#             continue
#
#         cell_names = sort_cell_keys(list(result.keys()))
#
#         print(f"Top-level cell keys: {cell_names[:30]}")
#
#         rmse_list = []
#         r2_list = []
#         wmape_list = []
#         mae_list = []
#
#         failed_cells = []
#
#         for name in cell_names:
#             try:
#                 rul_true = result[name][RUL_KEY][TRUE_KEY]
#                 rul_pred = result[name][RUL_KEY][PRED_KEY]
#             except Exception as e:
#                 failed_cells.append((name, str(e)))
#                 continue
#
#             metrics = compute_single_cell_metrics(rul_true, rul_pred)
#
#             if metrics is None:
#                 failed_cells.append((name, "invalid metric"))
#                 continue
#
#             rmse_list.append(metrics["RMSE"])
#             r2_list.append(metrics["R2"])
#             wmape_list.append(metrics["WMAPE"])
#             mae_list.append(metrics["MAE"])
#
#             detail_rows.append({
#                 "soc_range": f"{soc_end}%",
#                 "soc_end": soc_end,
#                 "cell_name": name,
#                 "RMSE": metrics["RMSE"],
#                 "R2": metrics["R2"],
#                 "WMAPE": metrics["WMAPE"],
#                 "MAE": metrics["MAE"],
#                 "n_points": metrics["n_points"]
#             })
#
#         data["RMSE"][soc_end] = np.array(rmse_list, dtype=float)
#         data["R2"][soc_end] = np.array(r2_list, dtype=float)
#         data["WMAPE"][soc_end] = np.array(wmape_list, dtype=float)
#         data["MAE"][soc_end] = np.array(mae_list, dtype=float)
#
#         print(f"Valid cells: {len(rmse_list)}")
#         print(f"Failed cells: {len(failed_cells)}")
#
#         if len(failed_cells) > 0:
#             print("[Warning] Failed cells preview:")
#             for item in failed_cells[:10]:
#                 print("  ", item)
#
#         for metric_name, values in [
#             ("RMSE", rmse_list),
#             ("R2", r2_list),
#             ("WMAPE", wmape_list),
#             ("MAE", mae_list)
#         ]:
#             arr = np.array(values, dtype=float)
#             arr = arr[np.isfinite(arr)]
#
#             if len(arr) == 0:
#                 print(f"[Warning] {metric_name}: no valid values")
#             else:
#                 print(
#                     f"{metric_name}: n={len(arr)}, "
#                     f"mean={np.mean(arr):.6f}, "
#                     f"std={np.std(arr, ddof=1) if len(arr) > 1 else 0.0:.6f}"
#                 )
#
#         summary_rows.append({
#             "soc_range": f"{soc_end}%",
#             "soc_end": soc_end,
#             "RMSE_mean": np.nanmean(rmse_list) if len(rmse_list) else np.nan,
#             "RMSE_std": np.nanstd(rmse_list, ddof=1) if len(rmse_list) > 1 else 0.0,
#             "R2_mean": np.nanmean(r2_list) if len(r2_list) else np.nan,
#             "R2_std": np.nanstd(r2_list, ddof=1) if len(r2_list) > 1 else 0.0,
#             "WMAPE_mean": np.nanmean(wmape_list) if len(wmape_list) else np.nan,
#             "WMAPE_std": np.nanstd(wmape_list, ddof=1) if len(wmape_list) > 1 else 0.0,
#             "MAE_mean": np.nanmean(mae_list) if len(mae_list) else np.nan,
#             "MAE_std": np.nanstd(mae_list, ddof=1) if len(mae_list) > 1 else 0.0,
#             "valid_cell_count": len(rmse_list)
#         })
#
#     return data, detail_rows, summary_rows
#
#
# # ============================================================
# # 保存 CSV
# # ============================================================
# def save_metric_detail_csv(detail_rows, save_path):
#     df = pd.DataFrame(detail_rows)
#     df.to_csv(save_path, index=False, encoding="utf-8-sig")
#     print(f"Detail metrics saved to: {save_path}")
#
#
# def save_metric_summary_csv(summary_rows, save_path):
#     df = pd.DataFrame(summary_rows)
#     df = df.sort_values("soc_end", ascending=False)
#     df.to_csv(save_path, index=False, encoding="utf-8-sig")
#     print(f"Summary metrics saved to: {save_path}")
#
#
# # ============================================================
# # 云雨图绘制函数
# # ============================================================
# def set_five_y_ticks(ax, values):
#     values = np.asarray(values, dtype=float)
#     values = values[np.isfinite(values)]
#
#     if len(values) == 0:
#         return
#
#     y_min = float(np.min(values))
#     y_max = float(np.max(values))
#
#     if y_min == y_max:
#         y_ticks = np.linspace(y_min - 1.0, y_max + 1.0, 5)
#     else:
#         y_pad = 0.08 * (y_max - y_min)
#         y_ticks = np.linspace(y_min - y_pad, y_max + y_pad, 5)
#
#     ax.set_yticks(y_ticks)
#
#
# def draw_half_violin(ax, values, position, color, width=0.65, alpha=0.35):
#     """
#     绘制右半边小提琴图。
#     """
#     if len(values) < 2:
#         return
#
#     parts = ax.violinplot(
#         [values],
#         positions=[position],
#         widths=width,
#         showmeans=False,
#         showmedians=False,
#         showextrema=False
#     )
#
#     for body in parts["bodies"]:
#         verts = body.get_paths()[0].vertices
#         verts[:, 0] = np.maximum(verts[:, 0], position)
#         body.set_facecolor(color)
#         body.set_edgecolor(color)
#         body.set_alpha(alpha)
#         body.set_linewidth(0.8)
#
#
# def draw_raincloud_single_metric(
#     ax,
#     values_by_soc,
#     soc_order,
#     metric_name,
#     ylabel,
#     rng,
#     colors
# ):
#     """
#     单个指标的云雨图。
#     """
#     all_values_for_ticks = []
#
#     for i, soc_end in enumerate(soc_order):
#         values = values_by_soc.get(soc_end, np.array([], dtype=float))
#         values = np.asarray(values, dtype=float)
#         values = values[np.isfinite(values)]
#
#         if len(values) == 0:
#             continue
#
#         all_values_for_ticks.extend(values.tolist())
#
#         pos = i + 1
#         color = colors[i % len(colors)]
#
#         # 云：右半小提琴
#         draw_half_violin(
#             ax=ax,
#             values=values,
#             position=pos,
#             color=color,
#             width=0.70,
#             alpha=0.30
#         )
#
#         # 雨：散点
#         jitter = rng.normal(loc=pos - 0.16, scale=0.035, size=len(values))
#
#         ax.scatter(
#             jitter,
#             values,
#             s=18,
#             color=color,
#             alpha=0.55,
#             edgecolors="none",
#             zorder=3
#         )
#
#         # 箱线图
#         box = ax.boxplot(
#             values,
#             positions=[pos - 0.03],
#             widths=0.14,
#             patch_artist=True,
#             showfliers=False,
#             zorder=4
#         )
#
#         for patch in box["boxes"]:
#             patch.set_facecolor("white")
#             patch.set_edgecolor("black")
#             patch.set_linewidth(1.0)
#
#         for median in box["medians"]:
#             median.set_color("black")
#             median.set_linewidth(1.4)
#
#         for whisker in box["whiskers"]:
#             whisker.set_color("black")
#             whisker.set_linewidth(1.0)
#
#         for cap in box["caps"]:
#             cap.set_color("black")
#             cap.set_linewidth(1.0)
#
#         # # 均值点
#         # ax.scatter(
#         #     pos + 0.18,
#         #     np.mean(values),
#         #     s=38,
#         #     color="black",
#         #     marker="D",
#         #     zorder=5
#         # )
#
#     ax.set_title(metric_name, fontsize=24)
#     ax.set_ylabel(ylabel, fontsize=24)
#     ax.set_xticks(np.arange(1, len(soc_order) + 1))
#     ax.set_xticklabels(
#         [f"0-{x}%" for x in soc_order],
#         fontsize=14,
#         rotation=35,
#         ha="center"
#     )
#
#     ax.grid(False)
#     ax.tick_params(axis="both", labelsize=20)
#
#     if len(all_values_for_ticks) > 0:
#         set_five_y_ticks(ax, np.array(all_values_for_ticks))
#
#     if metric_name.upper() == "R2":
#         ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#     elif metric_name.upper() == "WMAPE":
#         ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#     elif metric_name.upper() == "RMSE":
#         ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
#     else:
#         ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
#
#
# def get_soc_order(data, descending=True):
#     soc_order = sorted(
#         list({
#             soc_end
#             for metric in data
#             for soc_end in data[metric].keys()
#         }),
#         reverse=descending
#     )
#
#     return soc_order
#
#
# def plot_metric_rainclouds(data, save_dir):
#     """
#     绘制 2×2 云雨图。
#     """
#     soc_order = get_soc_order(data, descending=True)
#
#     colors = [
#         "#4C9FBF",
#         "#6BBF8A",
#         "#F2B84B",
#         "#E67E73",
#         "#9B7EBD",
#         "#7F8C8D",
#     ]
#
#     ylabel_map = {
#         "RMSE": "RMSE (cycles)",
#         "R2": "R2",
#         "WMAPE": "WMAPE",
#         "MAE": "MAE (cycles)",
#     }
#
#     metric_order = ["RMSE", "R2", "WMAPE", "MAE"]
#
#     rng = np.random.default_rng(1029)
#
#     fig, axes = plt.subplots(2, 2, figsize=(12, 9))
#     axes = axes.flatten()
#
#     for ax, metric in zip(axes, metric_order):
#         draw_raincloud_single_metric(
#             ax=ax,
#             values_by_soc=data[metric],
#             soc_order=soc_order,
#             metric_name=metric,
#             ylabel=ylabel_map[metric],
#             rng=rng,
#             colors=colors
#         )
#
#     plt.tight_layout()
#
#     save_tif = os.path.join(save_dir, "SOC_metric_rainclouds.tif")
#
#
#     plt.savefig(save_tif, dpi=300, bbox_inches="tight")
#
#     plt.show()
#
#     print(f"Figure saved to: {save_tif}")
#
#
#
# def plot_each_metric_separately(data, save_dir):
#     """
#     每个指标单独保存一张云雨图。
#     """
#     soc_order = get_soc_order(data, descending=True)
#
#     colors = [
#         "#4C9FBF",
#         "#6BBF8A",
#         "#F2B84B",
#         "#E67E73",
#         "#9B7EBD",
#         "#7F8C8D",
#     ]
#
#     ylabel_map = {
#         "RMSE": "RMSE (cycles)",
#         "R2": "R2",
#         "WMAPE": "WMAPE",
#         "MAE": "MAE (cycles)",
#     }
#
#     rng = np.random.default_rng(1029)
#
#     for metric in ["RMSE", "R2", "WMAPE", "MAE"]:
#         fig, ax = plt.subplots(figsize=(8, 6))
#
#         draw_raincloud_single_metric(
#             ax=ax,
#             values_by_soc=data[metric],
#             soc_order=soc_order,
#             metric_name=metric,
#             ylabel=ylabel_map[metric],
#             rng=rng,
#             colors=colors
#         )
#
#         ax.set_xlabel("SOC input range", fontsize=24)
#
#         plt.tight_layout()
#
#         save_tif = os.path.join(save_dir, f"SOC_{metric}_raincloud.tif")
#
#
#         plt.savefig(save_tif, dpi=300, bbox_inches="tight")
#
#         plt.show()
#
#         print(f"Figure saved to: {save_tif}")
#
#
#
# # ============================================================
# # 主程序
# # ============================================================
# if __name__ == "__main__":
#     data, detail_rows, summary_rows = collect_soc_metric_data(
#         result_dir=RESULT_DIR,
#         file_pattern=FILE_PATTERN
#     )
#
#
#     plot_metric_rainclouds(data, SAVE_DIR)
#     plot_each_metric_separately(data, SAVE_DIR)
#
#     print("\nAll SOC metric raincloud figures have been generated.")




import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR = r" "
SAVE_DIR = r" "
os.makedirs(SAVE_DIR, exist_ok=True)

FILE_PATTERN = " "

RUL_KEY = "rul"
TRUE_KEY = "true"
PRED_KEY = "base"

def load_obj_by_path(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def parse_soc_from_filename(file_path):
    filename = os.path.basename(file_path)
    match = re.search(r"SOC_0_(\d+)", filename)

    if match is None:
        return None

    return int(match.group(1))

def cal_wmape(predict, groundtruth):
    predict = np.asarray(predict, dtype=float).reshape(-1)
    groundtruth = np.asarray(groundtruth, dtype=float).reshape(-1)

    n = min(len(predict), len(groundtruth))
    predict = predict[:n]
    groundtruth = groundtruth[:n]

    mask = np.isfinite(predict) & np.isfinite(groundtruth)
    predict = predict[mask]
    groundtruth = groundtruth[mask]

    if len(groundtruth) == 0:
        return np.nan

    total_value = np.sum(np.abs(groundtruth))

    if total_value < 1e-12:
        return np.nan

    abs_error = np.abs(predict - groundtruth)

    return np.sum(abs_error) / total_value

def compute_single_cell_metrics(rul_true, rul_pred):
    rul_true = np.asarray(rul_true, dtype=float).reshape(-1)
    rul_pred = np.asarray(rul_pred, dtype=float).reshape(-1)

    n = min(len(rul_true), len(rul_pred))

    if n < 2:
        return None

    rul_true = rul_true[:n]
    rul_pred = rul_pred[:n]

    mask = np.isfinite(rul_true) & np.isfinite(rul_pred)
    rul_true = rul_true[mask]
    rul_pred = rul_pred[mask]

    if len(rul_true) < 2:
        return None

    rmse = float(np.sqrt(mean_squared_error(rul_true, rul_pred)))
    r2 = float(r2_score(rul_true, rul_pred))
    wmape = float(cal_wmape(rul_pred, rul_true))
    mae = float(mean_absolute_error(rul_true, rul_pred))

    return {
        "RMSE": rmse,
        "R2": r2,
        "WMAPE": wmape,
        "MAE": mae,
        "n_points": len(rul_true)
    }

def sort_cell_keys(cell_keys):

    def key_func(x):
        nums = re.findall(r"\d+", str(x))

        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])

        if len(nums) == 1:
            return int(nums[0]), 0

        return 9999, 9999

    return sorted(cell_keys, key=key_func)

def collect_soc_metric_data(result_dir, file_pattern):
    file_list = sorted(glob.glob(os.path.join(result_dir, file_pattern)))

    if len(file_list) == 0:
        raise FileNotFoundError(f"没有找到文件: {os.path.join(result_dir, file_pattern)}")

    data = {
        "RMSE": {},
        "R2": {},
        "WMAPE": {},
        "MAE": {}
    }

    detail_rows = []
    summary_rows = []

    for file_path in file_list:
        soc_end = parse_soc_from_filename(file_path)

        if soc_end is None:
            print(f"[Warning] 无法从文件名解析输入保留比例，跳过: {file_path}")
            continue

        result = load_obj_by_path(file_path)

        print("=" * 100)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Retained input proportion: {soc_end}%")

        if not isinstance(result, dict):
            print("[Warning] 当前文件内容不是 dict，跳过。")
            continue

        cell_names = sort_cell_keys(list(result.keys()))

        print(f"Top-level cell keys: {cell_names[:30]}")

        rmse_list = []
        r2_list = []
        wmape_list = []
        mae_list = []

        failed_cells = []

        for name in cell_names:
            try:
                rul_true = result[name][RUL_KEY][TRUE_KEY]
                rul_pred = result[name][RUL_KEY][PRED_KEY]
            except Exception as e:
                failed_cells.append((name, str(e)))
                continue

            metrics = compute_single_cell_metrics(rul_true, rul_pred)

            if metrics is None:
                failed_cells.append((name, "invalid metric"))
                continue

            rmse_list.append(metrics["RMSE"])
            r2_list.append(metrics["R2"])
            wmape_list.append(metrics["WMAPE"])
            mae_list.append(metrics["MAE"])

            detail_rows.append({
                "retained_input_proportion": f"{soc_end}%",
                "soc_end": soc_end,
                "cell_name": name,
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "WMAPE": metrics["WMAPE"],
                "MAE": metrics["MAE"],
                "n_points": metrics["n_points"]
            })

        data["RMSE"][soc_end] = np.array(rmse_list, dtype=float)
        data["R2"][soc_end] = np.array(r2_list, dtype=float)
        data["WMAPE"][soc_end] = np.array(wmape_list, dtype=float)
        data["MAE"][soc_end] = np.array(mae_list, dtype=float)

        print(f"Valid cells: {len(rmse_list)}")
        print(f"Failed cells: {len(failed_cells)}")

        if len(failed_cells) > 0:
            print("[Warning] Failed cells preview:")
            for item in failed_cells[:10]:
                print("  ", item)

        for metric_name, values in [
            ("RMSE", rmse_list),
            ("R2", r2_list),
            ("WMAPE", wmape_list),
            ("MAE", mae_list)
        ]:
            arr = np.array(values, dtype=float)
            arr = arr[np.isfinite(arr)]

            if len(arr) == 0:
                print(f"[Warning] {metric_name}: no valid values")
            else:
                print(
                    f"{metric_name}: n={len(arr)}, "
                    f"mean={np.mean(arr):.6f}, "
                    f"std={np.std(arr, ddof=1) if len(arr) > 1 else 0.0:.6f}"
                )

        summary_rows.append({
            "retained_input_proportion": f"{soc_end}%",
            "soc_end": soc_end,
            "RMSE_mean": np.nanmean(rmse_list) if len(rmse_list) else np.nan,
            "RMSE_std": np.nanstd(rmse_list, ddof=1) if len(rmse_list) > 1 else 0.0,
            "R2_mean": np.nanmean(r2_list) if len(r2_list) else np.nan,
            "R2_std": np.nanstd(r2_list, ddof=1) if len(r2_list) > 1 else 0.0,
            "WMAPE_mean": np.nanmean(wmape_list) if len(wmape_list) else np.nan,
            "WMAPE_std": np.nanstd(wmape_list, ddof=1) if len(wmape_list) > 1 else 0.0,
            "MAE_mean": np.nanmean(mae_list) if len(mae_list) else np.nan,
            "MAE_std": np.nanstd(mae_list, ddof=1) if len(mae_list) > 1 else 0.0,
            "valid_cell_count": len(rmse_list)
        })

    return data, detail_rows, summary_rows

def save_metric_detail_csv(detail_rows, save_path):
    df = pd.DataFrame(detail_rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Detail metrics saved to: {save_path}")


def save_metric_summary_csv(summary_rows, save_path):
    df = pd.DataFrame(summary_rows)
    df = df.sort_values("soc_end", ascending=False)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Summary metrics saved to: {save_path}")

def set_five_y_ticks(ax, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    y_min = float(np.min(values))
    y_max = float(np.max(values))

    if y_min == y_max:
        y_ticks = np.linspace(y_min - 1.0, y_max + 1.0, 5)
    else:
        y_pad = 0.08 * (y_max - y_min)
        y_ticks = np.linspace(y_min - y_pad, y_max + y_pad, 5)

    ax.set_yticks(y_ticks)

def draw_half_violin(ax, values, position, color, width=0.65, alpha=0.35):

    if len(values) < 2:
        return

    parts = ax.violinplot(
        [values],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body in parts["bodies"]:
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.maximum(verts[:, 0], position)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)
        body.set_linewidth(0.8)

def draw_raincloud_single_metric(
    ax,
    values_by_soc,
    soc_order,
    metric_name,
    ylabel,
    rng,
    colors
):

    all_values_for_ticks = []

    for i, soc_end in enumerate(soc_order):
        values = values_by_soc.get(soc_end, np.array([], dtype=float))
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        all_values_for_ticks.extend(values.tolist())

        pos = i + 1
        color = colors[i % len(colors)]

        draw_half_violin(
            ax=ax,
            values=values,
            position=pos,
            color=color,
            width=0.70,
            alpha=0.30
        )

        jitter = rng.normal(loc=pos - 0.16, scale=0.035, size=len(values))

        ax.scatter(
            jitter,
            values,
            s=18,
            color=color,
            alpha=0.55,
            edgecolors="none",
            zorder=3
        )

        box = ax.boxplot(
            values,
            positions=[pos - 0.03],
            widths=0.14,
            patch_artist=True,
            showfliers=False,
            zorder=4
        )

        for patch in box["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)

        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)

        for whisker in box["whiskers"]:
            whisker.set_color("black")
            whisker.set_linewidth(1.0)

        for cap in box["caps"]:
            cap.set_color("black")
            cap.set_linewidth(1.0)

    ax.set_title(metric_name, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)

    ax.set_xticks(np.arange(1, len(soc_order) + 1))

    ax.set_xticklabels(
        [f"{x}%" for x in soc_order],
        fontsize=14,
        rotation=35,
        ha="center"
    )

    ax.grid(False)
    ax.tick_params(axis="both", labelsize=20)

    if len(all_values_for_ticks) > 0:
        set_five_y_ticks(ax, np.array(all_values_for_ticks))

    if metric_name.upper() == "R2":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    elif metric_name.upper() == "WMAPE":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    elif metric_name.upper() == "RMSE":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))


def get_soc_order(data, descending=True):
    soc_order = sorted(
        list({
            soc_end
            for metric in data
            for soc_end in data[metric].keys()
        }),
        reverse=descending
    )

    return soc_order


def plot_metric_rainclouds(data, save_dir):

    soc_order = get_soc_order(data, descending=True)

    colors = [
        "#4C9FBF",
        "#6BBF8A",
        "#F2B84B",
        "#E67E73",
        "#9B7EBD",
        "#7F8C8D",
    ]

    ylabel_map = {
        "RMSE": "RMSE (cycles)",
        "R2": "R2",
        "WMAPE": "WMAPE",
        "MAE": "MAE (cycles)",
    }

    metric_order = ["RMSE", "R2", "WMAPE", "MAE"]

    rng = np.random.default_rng(1029)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metric_order):
        draw_raincloud_single_metric(
            ax=ax,
            values_by_soc=data[metric],
            soc_order=soc_order,
            metric_name=metric,
            ylabel=ylabel_map[metric],
            rng=rng,
            colors=colors
        )

    plt.tight_layout()

    save_tif = os.path.join(save_dir, "SOC_metric_rainclouds.tif")

    plt.savefig(save_tif, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {save_tif}")


def plot_each_metric_separately(data, save_dir):

    soc_order = get_soc_order(data, descending=True)

    colors = [
        "#4C9FBF",
        "#6BBF8A",
        "#F2B84B",
        "#E67E73",
        "#9B7EBD",
        "#7F8C8D",
    ]

    ylabel_map = {
        "RMSE": "RMSE (cycles)",
        "R2": "R2",
        "WMAPE": "WMAPE",
        "MAE": "MAE (cycles)",
    }

    rng = np.random.default_rng(1029)

    for metric in ["RMSE", "R2", "WMAPE", "MAE"]:
        fig, ax = plt.subplots(figsize=(8, 6))

        draw_raincloud_single_metric(
            ax=ax,
            values_by_soc=data[metric],
            soc_order=soc_order,
            metric_name=metric,
            ylabel=ylabel_map[metric],
            rng=rng,
            colors=colors
        )

        ax.set_xlabel("Retained input proportion", fontsize=24)

        plt.tight_layout()

        save_tif = os.path.join(save_dir, f"SOC_{metric}_raincloud.tif")

        plt.savefig(save_tif, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Figure saved to: {save_tif}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    data, detail_rows, summary_rows = collect_soc_metric_data(
        result_dir=RESULT_DIR,
        file_pattern=FILE_PATTERN
    )

    detail_csv = os.path.join(SAVE_DIR, "SOC_metric_detail.csv")
    summary_csv = os.path.join(SAVE_DIR, "SOC_metric_summary.csv")

    save_metric_detail_csv(detail_rows, detail_csv)
    save_metric_summary_csv(summary_rows, summary_csv)

    plot_metric_rainclouds(data, SAVE_DIR)
    plot_each_metric_separately(data, SAVE_DIR)

    print("\nAll SOC metric raincloud figures have been generated.")