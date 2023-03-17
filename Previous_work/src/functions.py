import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt

def set_seed(seed):
    """
    Sets a global random seed of your choice
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_lists(data):
    """
    This function retrieves a full column list, four variables with the four different types of columns
    and a list of columns that contain at least one missing value.
    """
    col_list = data.columns.tolist()
    F1 = [col for col in data.columns if 'F_1' in col]
    F2 = [col for col in data.columns if 'F_2' in col]
    F3 = [col for col in data.columns if 'F_3' in col]
    F4 = [col for col in data.columns if 'F_4' in col]
    missing_cols = [col for col in data.columns if data[col].isnull().sum() != 0]
    return col_list, F1, F2, F3, F4, missing_cols

def clf_plot_distributions(data, features, hue='target', ncols=3, method='hist'):
    nrows = int(len(features) / ncols) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, round(nrows*16/ncols)))
    for ax,feature in zip(axes.ravel()[:len(features)],features):
        if method == 'hist':
            sns.kdeplot(data=data, x=feature, ax=ax)
        elif method == 'cdf':
            sns.ecdfplot(data=data, x=feature, ax=ax)
        elif method == 'box':
            sns.boxplot(data=data, x=feature, ax=ax)
        elif method == 'bar':
            temp = data.copy()
            temp['counts'] = 1
            temp = temp.groupby([feature], as_index=False).agg({'counts':'sum'})
            sns.barplot(data=temp, x=feature, y='counts', ax=ax)
        elif method == 'hbar':
            temp = data.copy()
            temp['counts'] = 1
            temp = temp.groupby([feature], as_index=False).agg({'counts':'sum'})
            sns.barplot(data=temp, y=feature, x='counts', ax=ax)
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()

def outlier_elimination(data, lower_bound=-2.5, upper_bound=2.5):
    """
    Define a max and min value for the features that were observed to have outliers.
    -2.5 and 2.5 should be the stringest limit.
    """
    data['F_1_7'] = data['F_1_7'].clip(lower_bound, upper_bound)
    data['F_1_12'] = data['F_1_12'].clip(lower_bound, upper_bound)
    data['F_1_13'] = data['F_1_13'].clip(lower_bound, upper_bound)
    data['F_3_19'] = data['F_3_19'].clip(lower_bound, upper_bound)
    data['F_3_21'] = data['F_3_21'].clip(lower_bound, upper_bound)
    data['F_4_2'] = data['F_4_2'].clip(lower_bound, upper_bound)
    data['F_4_3'] = data['F_4_3'].clip(lower_bound, upper_bound)
    data['F_4_8'] = data['F_4_8'].clip(lower_bound, upper_bound)
    data['F_4_9'] = data['F_4_9'].clip(lower_bound, upper_bound)
    data['F_4_10'] = data['F_4_10'].clip(lower_bound, upper_bound)
    data['F_4_14'] = data['F_4_14'].clip(lower_bound, upper_bound)
    return data

def get_column_na_index(df, column):
    """
    For use generator style in 'pandas_group_impute' to generate a list of the dataframe's indexes corresbonding with missing values, iterating through columns
    """
    return df[df[column].isna() == True].index

def na_nona_index_na_cnt(col, cnt):
    na_index = na_index_of_column[col]
    no_na_index = no_na_index_of_column[col]
    na_cnt_index = na_cnt_index_of[cnt]
    na_index = na_index.intersection(na_cnt_index)
    no_na_index = no_na_index.intersection(na_cnt_index)
    return na_index, no_na_index

def reduce_memory(data, verbose=True):
    """
    Goes through the dataframe and changes the data type to one that uses less memory if applicable
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = data.memory_usage().sum() / 1024 ** 2
    for column in data.columns:
        column_type = data[column].dtypes
        if column_type in numerics:
            c_min = data[column].min()
            c_max = data[column].max()
            if str(column_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[column] = data[column].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[column] = data[column].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[column] = data[column].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[column] = data[column].astype(np.int64)
            elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                data[column] = data[column].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                data[column] = data[column].astype(np.float32)
            else:
                data[column] = data[column].astype(np.float64)
    end_mem = data.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100*(start_mem - end_mem) / start_mem))
    return data

def make_submission(imputed_data, sample_submission):
    for i in sample_submission.index: 
        row = int(i.split('-')[0])
        col = i.split('-')[1]
        sample_submission.loc[i, 'value'] = imputed_data.loc[row, col]

    return sample_submission


