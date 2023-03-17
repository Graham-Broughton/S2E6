import numpy as np
import pandas as pd
import random
import os

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import backend as K

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

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def make_submission(imputed_data, sample_submission):
    for i in sample_submission.index: 
        row = int(i.split('-')[0])
        col = i.split('-')[1]
        sample_submission.loc[i, 'value'] = imputed_data.loc[row, col]

    return sample_submission