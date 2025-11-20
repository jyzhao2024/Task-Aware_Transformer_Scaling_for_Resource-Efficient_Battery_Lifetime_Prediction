import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from matplotlib.font_manager import FontProperties
import seaborn as sns
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


#
result = load_obj('')

test_1 = ['a9', 'a14', 'a17', 'a23']
#
fig = plt.figure(figsize=(10, 9))
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
x_lim = 1000
plt.plot(range(x_lim), range(x_lim), '-', c='k', linewidth=3, label='y=x')
cmap = plt.get_cmap('YlGnBu')
norm = plt.Normalize(vmin=0.3, vmax=1)
#
colors = ['#fabfbd', '#d4d58d', '#8be2c4', '#fb6295', '#9ca5de', '#8AD9F6', ]
#
#
font_prop = FontProperties(family='Times New Roman', size=18)#
#
label_names = test_1[:]
for i, name in enumerate(test_1):
    interval = 20

    rul_true = result[name]['rul']['true']
    rul_base = result[name]['rul']['base']


    color = [len(rul_true) / 2700] * len(rul_true)
    color = color[::interval]

    plt.plot(rul_true[::interval], rul_base[::interval], '^', markersize=8, label=label_names[i], c=colors[i])
plt.legend(fontsize=20, loc="lower right",prop=font_prop)
x_major_locator = MultipleLocator(400)
y_major_locator = MultipleLocator(400)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim((0, x_lim))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(length=10)
plt.xticks([0, 500, 1000], size=35,family='Times New Roman')
plt.yticks([0, 500, 1000], size=35, family='Times New Roman')

ax.set_xlabel('Observed cycle life (cycles)',fontsize=35, family='Times New Roman')
ax.set_ylabel('Predicted cycle life (cycles)',fontsize=35, family='Times New Roman')

plt.savefig('./rul33_1.tiff', dpi=800, format="tiff")


#
fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(left=0.15, right=0.9, top = 0.9, bottom = 0.15)
cmap = plt.get_cmap('YlGnBu')
norm = plt.Normalize(vmin=0.3, vmax=1)
error_list = []

for i, name in enumerate(test_1):
    interval = 1
    rul_true = result[name]['rul']['true']
    rul_base = result[name]['rul']['base']
    tmp = rul_base[::interval] - rul_true[::interval]
    error_list.append(tmp.reshape(-1, 1))
error_array = np.vstack(error_list)

sns.kdeplot(error_array.squeeze(), color='#75DBC5', fill=True)
plt.xticks([-100, 0, 100], size=35, family='Times New Roman')
plt.xlim(-250,250)
plt.yticks([], size=35, family='Times New Roman')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.ylabel('Frequency', fontsize=35, family='Times New Roman')
plt.savefig('.tiff', dpi=800, format="tiff")
plt.show()