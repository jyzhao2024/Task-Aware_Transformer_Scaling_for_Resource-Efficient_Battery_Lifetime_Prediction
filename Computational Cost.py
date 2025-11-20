import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
import os

#
file_path =  r''
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names
all_data = {sheet: xls.parse(sheet) for sheet in sheet_names}

#
max_length = 20
cycle_names = list(all_data.keys())

#
metric_data = {
    'Inference time': pd.DataFrame(index=range(max_length)),
    'Flops': pd.DataFrame(index=range(max_length)),
}

#
for cycle in cycle_names:
    df = all_data[cycle]
    for metric in metric_data:
        if metric in df.columns:
            metric_data[metric][cycle] = df[metric].iloc[:max_length].reindex(range(max_length)).ffill()


#
colors = ['#fabfbd', '#8be2c4', '#fb6295', '#9ca5de', '#8AD9F6']
cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)

#
font_prop = FontProperties(family='Times New Roman', size=25)
font_prop_legend = FontProperties(family='Times New Roman', size=15)

#
os.makedirs("Figure", exist_ok=True)

#
for metric, data in metric_data.items():
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    size_scale = 3000  #
    sampled_colors = [cmap(i / (len(data.columns) - 1)) for i in range(len(data.columns))]

    #
    for idx, col in enumerate(data.columns):
        y = data[col].values
        x = np.arange(len(y))
        max_val =  y.max()
        print(max_val)
        sizes = (y / max_val) * size_scale
        if metric == 'Inference time':
           ax.scatter(x, y, s=sizes, color=sampled_colors[idx], alpha=0.5,
                   label=col, edgecolors=sampled_colors[idx], linewidth=1)
        elif metric == 'Flops':

            ax.plot(x, y, color=sampled_colors[idx], linewidth=2.0)
            ax.fill_between(x, y, color=sampled_colors[idx], alpha=0.4)

    ax.set_xlabel("Data index", fontproperties=font_prop)
    ax.set_ylabel(metric, fontproperties=font_prop)
    ax.tick_params(labelsize=25)

    custom_labels = [
                        "Model parameters: 1.8619 M",
                        "Model parameters: 0.6197 M",
                        "Model parameters: 0.2320 M",
                        "Model parameters: 0.1058 M",
                        "Model parameters: 0.0374 M",
                        "Model parameters: 0.0205 M",
                        "Model parameters: 0.00996M",

                    ][:len(data.columns)]  #

    #
    mean_vals = data.mean().values
    scaled_sizes = (mean_vals / mean_vals.max()) * 100

    #
    legend_loc = 'upper right'

    #
    marker_size_factor =2

    #
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=sampled_colors[i],
                   markersize=np.sqrt(scaled_sizes[i]) * marker_size_factor,
                   label=custom_labels[i])
        for i in range(len(data.columns))
    ]

    #
    ax.legend(handles=legend_handles,
              loc=legend_loc,
              prop=font_prop_legend,
              title='Model size',
              title_fontproperties=font_prop_legend,
              framealpha=0.3,
              labelspacing=0.5,
              handlelength=1.0)
    #
    ax.set_ylim(0, data.max().max() * 1.1)
    ax.set_xticks([4, 9, 14, 19])
    ax.set_xticklabels(['5', '10', '15', '20'], fontproperties=font_prop)
    if metric == 'Inference time':
        ax.set_xlim(-1, max_length)
        ax.set_ylim(-0.05, data.max().max() * 1.1)
    else:
        ax.set_xlim(0, max_length-1)
        ax.set_ylim(0.01, data.max().max() * 1.1)

    #
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    #
    plt.tight_layout()
    plt.savefig(f'Figure/modelparameters_{metric}.tiff', dpi=800, format="tiff")
    plt.show()

#
file_path =  r'E:\zhao\FAST_tianjia\compute cost.xlsx'
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names
all_data = {sheet: xls.parse(sheet) for sheet in sheet_names}

#
max_length = 20
cycle_names = list(all_data.keys())

#
metric_data = {
    'Inference time': pd.DataFrame(index=range(max_length)),
    'Flops': pd.DataFrame(index=range(max_length)),
}

#
for cycle in cycle_names:
    df = all_data[cycle]
    for metric in metric_data:
        if metric in df.columns:
            metric_data[metric][cycle] = df[metric].iloc[:max_length].reindex(range(max_length)).ffill()

#
colors = [ '#fb6295','#9ca5de', '#8AD9F6']
cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)

#
font_prop = FontProperties(family='Times New Roman', size=35)
font_prop_legend = FontProperties(family='Times New Roman', size=18)

#
os.makedirs("Figure", exist_ok=True)

#
for metric, data in metric_data.items():
    if metric == 'Flops':
        fig, ax = plt.subplots(figsize=(6, 3))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        sampled_colors = [cmap(i / (len(data.columns) - 1)) for i in range(len(data.columns))]
        for idx, col in enumerate(data.columns):
            if idx >= 4:
                y = data[col].values
                x = np.arange(len(y))
                max_val = y.max()
                print(max_val)
                ax.plot(x, y, color=sampled_colors[idx], linewidth=2.0)
                ax.fill_between(x, y, color=sampled_colors[idx], alpha=0.4)


    ax.tick_params(labelsize=25)
    ax.set_xlim(0,19 )
    ax.set_ylim(0.01, )
    #
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    plt.tight_layout()
    plt.savefig(f'Figure/modelparameters_{metric}_zoom.tiff', dpi=800, format="tiff")
    plt.show()
