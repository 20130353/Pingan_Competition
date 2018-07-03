# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/4/24 9:30
# file: preproess.py
# description: 数据预处理文件

import pandas as pd
import numpy as np
import math


def save_long_trip(data):
    '''
    保留行程较长的trip
    :param data:
    :return:
    '''
    LONGITUDE1 = data['LONGITUDE']
    save_data1 = data.drop('LONGITUDE',axis=1)
    save_data1.insert(0,'LONGITUDE',LONGITUDE1)
    temp_data = save_data1.groupby(['TERMINALNO', 'TRIP_ID']).transform(lambda x:x.max()-x.min())
    save_index1 = temp_data[(temp_data.LONGITUDE > 0.036) | (temp_data.LONGITUDE < -0.036)].index.tolist()

    LATITUDE1 = data['LATITUDE']
    save_data2 = data.drop('LATITUDE', axis=1)
    save_data2.insert(0, 'LATITUDE', LATITUDE1)
    temp_data = save_data2.groupby(['TERMINALNO', 'TRIP_ID']).transform(lambda x: x.max() - x.min())
    save_index2 = temp_data[(temp_data.LATITUDE > 0.036) | (temp_data.LATITUDE < -0.036)].index.tolist()

    save_index = save_index1 + save_index2
    new_data = data.loc[save_index]

    return new_data

def process_mistake_missing_duplicates(data_df):
    """
    去除数据中的缺失值，错误值和重复值（删除时间后，去掉所有重复的）
    :param data_df:
    :return:
    """

    columns_name = ['TERMINALNO', 'TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    mean_value = data_df['SPEED'][data_df.SPEED != -1].mean()
    speed_df = pd.DataFrame(data_df['SPEED'].replace([-1], [mean_value]))  # 均值速度填充缺失值


    call_df = pd.DataFrame(data_df['CALLSTATE'].replace([-1, 2, 3, 4], [0, 1, 1, 0]))  # 修改callstate的状态为-1,2,3,4-没打电话，打电话，打电话和没打电话
    data_df = data_df[(73 < data_df.LONGITUDE) & (135 > data_df.LONGITUDE)] # 中国经度73~135之间
    data_df = data_df[(18 < data_df.LATITUDE) & (53 > data_df.LATITUDE)]  # 中国维度20~53之间
    data_df = data_df[data_df.DIRECTION != -1] # 删除方向丢失的数据
    data_df = data_df[(-150 < data_df.HEIGHT) & (6700 > data_df.HEIGHT)]  # 中国高度20~53之间

    if 'Y' in data_df.columns.tolist():
        no_repeat_data = data_df.drop_duplicates(columns_name.remove('TIME'))  # 去除重复数据
    else:
        columns_name.remove('TIME')
        columns_name.remove('Y')
        no_repeat_data = data_df.drop_duplicates(columns_name)  # 去除重复数据

    return no_repeat_data

def preproess_fun(train_df,test_df):
    """

    :param data: type-array
    :param label: type-array
    :return: data,label
    """
    save_long_trip(train_df)
    save_long_trip(test_df)

    columns_name = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED',
                    'CALLSTATE', 'Y']
    train_df = process_mistake_missing_duplicates(train_df)
    test_df = process_mistake_missing_duplicates(test_df)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler() # 归一化高度
    train_height = pd.DataFrame({'MINMAX_HEIGHT':scaler.fit_transform(train_df[['HEIGHT']])[:,0]})# train height
    test_height = pd.DataFrame({'MINMAX_HEIGHT': scaler.transform(test_df[['HEIGHT']])[:, 0]})#test height

    scaler = MinMaxScaler()# 归一化速度
    train_speed = pd.DataFrame({'MINMAX_SPEED': scaler.fit_transform(train_df[['SPEED']])[:, 0]})  # train height
    test_speed = pd.DataFrame({'MINMAX_SPEED': scaler.transform(test_df[['SPEED']])[:, 0]})  # test height

    train_new = pd.concat([train_height, train_speed, train_df[columns_name].reset_index(drop=True)], axis=1) # 修改好的数据拼接成原数据
    test_new = pd.concat([test_height, test_speed, test_df[columns_name[:9]].reset_index(drop=True) ], axis=1) # 修改好的数据拼接成原数据

    return train_new,test_new

