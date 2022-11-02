import os
import numpy as np
import pandas as pd
import math

from datetime import datetime
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import scipy
import matplotlib.pyplot as plt

import json

from . import config as c

def create_folder_for_output(parent_dir):
    """ create an output folder,
    to store all the outputs of the trained model:
    weights of models, metrics of models """

    def create_folder(folder_path):
        folder_path = os.path.abspath(folder_path)
        project_folder = os.path.dirname(os.getcwd())

        if os.path.commonpath([folder_path, project_folder]) != project_folder:
            return -1
        if folder_path == project_folder:
            return 0
        if os.path.isdir(folder_path):
            return 0
        create_folder(os.path.dirname(folder_path))
        os.mkdir(folder_path)
        return 0

    create_folder(os.path.join(parent_dir, 'output'))

    create_folder(os.path.join(parent_dir, 'output', 'config'))
    create_folder(os.path.join(parent_dir, 'output', 'img'))
    create_folder(os.path.join(parent_dir, 'output', 'lgbm_model'))
    create_folder(os.path.join(parent_dir, 'output', 'lstm_model'))
    create_folder(os.path.join(parent_dir, 'output', 'results'))

    create_folder(os.path.join(parent_dir, 'output', 'img', 'comp_graph'))
    create_folder(os.path.join(parent_dir, 'output', 'img', 'scatter_plot'))

    create_folder(os.path.join(parent_dir, 'output', 'lstm_model', 'loggers'))

def load_data(data_dir):
    """ read csv file data to dataframe """
    df = pd.read_csv(data_dir)
    return df

def generate_time_series_data(df, window_size, stride_pred):
    """ generate time series data with existing dataframe
    with number of data points in one sampling = window_size and
    prediction distance = stride_pred respectively """

    def decomposes_into_valid_sub_df_list():
        """ subdivide the parent df into a list of dataframes containing continuous data points
        discontinuous data points are adjacent points and the distance is > 1 hour """

        list_df = []
        mark = 0

        for i in range(1, len(df)):
            prev = datetime.strptime(df['time'][i - 1], '%Y-%m-%d %H:%S')
            curr = datetime.strptime(df['time'][i], '%Y-%m-%d %H:%S')

            if curr - prev != timedelta(hours=1):
                list_df.append(df.iloc[mark:i, :])
                mark = i

        return list_df

    # list of data frames containing continuous data
    sub_df_list = decomposes_into_valid_sub_df_list()
    # print(f'Before remove sub_df which has length <= `time_step+1`: {len(sub_df_list)}')

    # filter out a list of data frames of suitable length to sample the data
    sub_df_list = [ df_i for df_i in sub_df_list if len(df_i) >= window_size + stride_pred ]
    # print(f'After remove sub_df which has length >= `time_step+1`: {len(sub_df_list)}')


    # X_training, y_training list
    X, y = [], []

    # list columns of dataframe
    feature_num = len(c.features_list)

    for df_i in sub_df_list:
        # list contain all feature in dataframe, except for the feature: `time`
        df_i = df_i[c.features_list]
        # convert dataframe to numpy array
        df_i = df_i.values.reshape(-1, feature_num)
        # print('len df_i:', len(df_i))

        # create sample for data
        s, e = 0, len(df_i) - stride_pred - window_size + 1
        # print(s, " :", e)
        for i in range(s, e):
            X.append(df_i[i: i+window_size])
            y.append(df_i[i+window_size+stride_pred-1][-1])

    X, y = np.array(X), np.array(y)

    # # check shape X, y
    # print(f'Shape: X{X.shape}, y{y.shape}')
    # # check first sample ~ first column in csv
    # print(X[0][-1], y[0])
    # # check last sample ~ last column in csv
    # print(X[-1][-1], y[-1])

    return X, y

def normalize_data(X):
    """ Min Max Scaler for time series data in np.array format,
     have shape (length, window_size, feature_num) """
    # auto window_size, feature_num
    ws, fn = X[0].shape
    sc = MinMaxScaler()

    X = sc.fit_transform(X.reshape(-1, ws * fn))
    X = X.reshape(-1, ws, fn)
    return X

def split_data(X, y, train_ratio):
    split_point = round(len(X) * train_ratio)
    x_train, x_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    return x_train, y_train, x_test, y_test