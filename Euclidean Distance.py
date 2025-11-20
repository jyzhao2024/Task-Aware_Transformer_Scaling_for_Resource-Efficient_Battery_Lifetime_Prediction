import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial.distance import cdist
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
#
def load_obj(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


#
def extract_discharge_capacity(battery_names, base_path):
    capacity_data, cycle_data, battery_indices = [], [], []

    for bat_num in battery_names:
        battery_data = load_obj(f'{base_path}/{bat_num}.pkl')[bat_num]
        A_dq = battery_data['dq']

        for cycle, dq_value in A_dq.items():
            if cycle>10:
                if isinstance(dq_value, (int, float)):
                    capacity_data.append(dq_value / 1000)
                    cycle_data.append(cycle)
                    # print( cycle_data)
                    battery_indices.append(bat_num)

    unique_batteries = list(set(battery_indices))
    battery_indices = np.array([unique_batteries.index(bat) + 1 for bat in battery_indices])

    return np.array(cycle_data, dtype=int), battery_indices, np.array(capacity_data, dtype=float)


#
def extract_charge_capacity(battery_names, base_path):
    charge_capacity_data, voltage_data, cycle_data = [], [], []

    for bat_num in battery_names:
        battery_data = load_obj(f'{base_path}/{bat_num}.pkl')[bat_num]
        A_dq = battery_data['dq']
        A_df = battery_data['data']
        all_idx = list(A_dq.keys())[9:]  #
        q, v, cycle = [], [],[]

        for cyc in all_idx:
            tmp = A_df[cyc]
            tmp = tmp.loc[tmp['Status'].apply(lambda x: not 'discharge' in x)]
            tmp_v = tmp['Voltage (V)'].values
            tmp_q = tmp['Capacity (mAh)'].values
            #
            cycle.append(cyc)
            q.extend(tmp_q)
            v.extend(tmp_v)


        charge_capacity_data.append(float(np.mean(q)))
        voltage_data.append(float(np.mean(v)))
        cycle_data.append(float(np.mean(cycle)))
        # print(charge_capacity_data.shape,voltage_data.shape,cycle_data.shape)

    return (
        np.array(charge_capacity_data, dtype=float),
        np.array(voltage_data, dtype=float),
        np.array(cycle_data, dtype=int)
    )



#
def normalize_data(data):
    data = np.array(data)
    if data.max() == data.min():  #
        return np.zeros_like(data)
    return (data - data.min()) / (data.max() - data.min())


#
def calculate_similarity(train_features, test_features):
    return cdist(train_features, test_features, metric='euclidean')


def plot_3d_with_colorbar(ncm_indices, lfp_indices, similarities):
    #
    sample_indices = range(0, len(lfp_indices), 1)
    lfp_indices = lfp_indices[sample_indices]
    ncm_indices = ncm_indices[sample_indices]
    similarities = similarities[sample_indices]

    fig = plt.figure(figsize=(12, 8))  #
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#B9274B','#CE6065', '#EBE991', '#9ECE9A', '#1574A5','#516199']  #
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    #
    sc = ax.scatter(
        lfp_indices,
        ncm_indices,
        similarities,
        c=similarities,
        cmap=cmap,  #
        s=similarities * 80,  #
        alpha=0.9#
    )

    font = {'family': 'Times New Roman', 'size': 25}
    plt.rc('font', **font)

    #
    font_prop = FontProperties(family='Times New Roman', size=25)
    ax.tick_params(axis='both', which='major', labelsize=25)
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontproperties(font_prop)

    #
    ax.set_ylabel('Train battery', labelpad=15, fontsize=25, fontname='Times New Roman')
    ax.set_xlabel('Test battery', labelpad=15, fontsize=25, fontname='Times New Roman')
    ax.set_zlabel('Euclidean distance', labelpad=15, fontsize=25, fontname='Times New Roman')

    ax.xaxis.set_major_locator(MultipleLocator(5))  #
    ax.yaxis.set_major_locator(MultipleLocator(8))  #
    ax.zaxis.set_major_locator(MultipleLocator(0.2))  #

    #
    ax.view_init(elev=30, azim=60)  #

    #
    cbar = plt.colorbar(sc, pad=0.05, shrink=0.7)
    cbar.ax.tick_params(labelsize=28)  # 设置色条刻度字号

    plt.show()


def plot_3d_combined_curve(train_x, train_y, train_z, test_x, test_y, test_z):  #
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_train_batteries = np.unique(train_y)
    unique_test_batteries = np.unique(test_y)

    #
    colors = ['#B9274B', '#CE6065', '#EBE991', '#9ECE9A', '#1574A5', '#516199']
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    #
    num_batteries = max(len(unique_test_batteries), len(unique_train_batteries))
    color_indices = np.linspace(0, 1, num_batteries)  #
    color_map_values = [cmap(ci) for ci in color_indices]  #

    spacing_factor = 10 #

    for i, battery in enumerate(unique_train_batteries):
        mask = train_y == battery
        ax.plot(train_y[mask]* spacing_factor, train_x[mask], train_z[mask], linestyle='-', color=color_map_values[i],
                label=f'Train {battery}' if i == 0 else "")

    #
    for i, battery in enumerate(unique_test_batteries):
        mask = test_y == battery
        ax.plot(test_y[mask]* spacing_factor, test_x[mask], test_z[mask], linestyle='--',#
                color=color_map_values[i], label=f'Test {battery}' if i == 0 else "")

    #
    ax.set_ylabel('Cycle number', labelpad=15, fontsize=25, fontname='Times New Roman')
    ax.set_xlabel('Battery index', labelpad=15,fontsize=25, fontname='Times New Roman')
    ax.set_zlabel('Discharge capacity (Ah)', labelpad=15,fontsize=25, fontname='Times New Roman')

    #
    font = {'family': 'Times New Roman', 'size': 25}
    plt.rc('font', **font)

    #
    font_prop = FontProperties(family='Times New Roman', size=25)
    ax.tick_params(axis='both', which='major', labelsize=25)
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontproperties(font_prop)

    ax.view_init(elev=30, azim=225)  #
    ax.set_ylim(bottom=max(train_x.max(), test_x.max()), top=0)#从


    plt.show()



#
def main(base_path, new_train, new_test):
    #
    train_charge_cap, train_voltage, train_cycles = extract_charge_capacity(new_train, base_path)
    test_charge_cap, test_voltage, test_cycles = extract_charge_capacity(new_test, base_path)

    train_charge_cap = normalize_data(train_charge_cap)
    test_charge_cap = normalize_data(test_charge_cap)
    train_cycles= normalize_data(train_cycles)
    train_voltage = normalize_data(train_voltage)
    test_voltage = normalize_data(test_voltage)
    test_cycles=normalize_data(test_cycles)
    train_points = np.array(list(zip(train_voltage, train_charge_cap, train_cycles)))
    print( train_points)
    test_points = np.array(list(zip(test_voltage, test_charge_cap, test_cycles)))
    print(f"Train Features Shape: {train_points.shape}")
    print(f"Test Features Shape: {test_points.shape}")
    #
    similarity_matrix = calculate_similarity(train_points, test_points)
    print( similarity_matrix )

    #
    train_indices = np.arange(len(train_points))
    test_indices = np.arange(len(test_points))
    train_indices, test_indices = np.meshgrid(train_indices, test_indices, indexing='ij')
    print(train_indices, test_indices )

    similarities = similarity_matrix.flatten()
    similarities =normalize_data( similarities)
#

if __name__ == "__main__":
    base_path = "./our_data"
    all_loader = load_obj(f"{base_path}/all_loader.pkl")

    random.seed(20250220)
    keys = list(all_loader.keys())
    random.shuffle(keys)

    new_train, new_test = keys[:10]+keys[47:57], keys[57:]
    main(base_path,  new_train, new_test)
