import pickle
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
result = load_obj('./result/****')
print(result.keys())

new_test = result.keys()
test_channel = ['#' + i[1:] for i in new_test]
test_nm_ch = set(zip(test_channel, new_test))

sort_new_test = list(test_nm_ch)
test_name = [i[0] for i in sort_new_test]
test_channel = [i[1] for i in sort_new_test]

cols = ['rmse_before', 'r2_before', 'wmape_before', 'mae_before']
table = pd.DataFrame(index=[test_name, test_channel], columns=cols)
table.loc['All', :] = 0

def cal_wmape(predict, groundtruth):
    abs_error = np.abs(predict - groundtruth)
    total_value = np.sum(groundtruth)
    return np.sum(abs_error) / total_value

rmse_before_list, r2_before_list, mae_before_list, wmape_before_list = [], [], [], []

for (code, name) in sort_new_test[:]:
    rul_true = result[name]['rul']['true']
    rul_base = result[name]['rul']['base']         
  
    rmse_before = np.sqrt(mean_squared_error(rul_true, rul_base))
    r2_before = r2_score(rul_true, rul_base)
    wmape_before = cal_wmape(rul_base, rul_true)
    mae_before = mean_absolute_error(rul_true, rul_base)

    table.loc[(code, name), ['rmse_before']] = ['%.3g' % rmse_before]
    table.loc[(code, name), ['r2_before']] = ['%.3g' % r2_before]
    table.loc[(code, name), ['wmape_before']] = ['%.3g' % wmape_before]
    table.loc[(code, name), ['mae_before']] = ['%.3g' % mae_before]

    rmse_before_list.append(rmse_before)
    r2_before_list.append(r2_before)
    wmape_before_list.append(wmape_before)
    mae_before_list.append(mae_before)

table.loc['All', ['rmse_before']] = ['%.3g' % np.mean(rmse_before_list)]
table.loc['All', ['r2_before']] = ['%.3g' % np.mean(r2_before_list)]
table.loc['All', ['wmape_before']] = ['%.3g' % np.mean(wmape_before_list)]
table.loc['All', ['mae_before']] = ['%.3g' % np.mean(mae_before_list)]

pd.set_option('display.max_rows', None)
print(table.to_string(index=False))
