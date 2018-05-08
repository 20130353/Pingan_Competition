# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import time
import random
import numpy as py
import math

import creat_features

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

def write_result(id,pre_lable):
    """

    :param id:type-series
    :param pre_lable:type-array
    :return:nothing
    """
    dataframe = pd.DataFrame({'Id': id, 'Pred': pre_lable}, dtype=float)
    dataframe = pd.pivot_table(dataframe, index=['Id'])
    dataframe.to_csv("model/test.csv", index=True, sep=',')

def print_correlation(data,options):
    columns_name = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED',
                    'CALLSTATE', 'Y']

    features_name = ['CALLSTATE','MINMAX_SPEED', 'MINMAX_HEIGHT','TOTAL_RUN', 'MEAN_RUN',\
                     'UNSAFE_SPEED_R', 'UNSAFE_TRIP_R','UNSAFE_START_SPEED_R','UNSAFE_START_TRIP_R']

    # for index in columns_name:
    #     first = str(features_name[index]) + ' -- y cor:\t'
    #     second = str(round(data[[features_name[index], 'Y']].corr().values[0, 1], 2))
    #     print(first + second)

    for index in options:
        first = str(features_name[index]) + ' -- y cor:\t'
        second = str(round(data[[features_name[index], 'Y']].corr().values[0, 1],2))
        print(first + second)


def strategy(data):

    columns_name = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED',
                    'CALLSTATE', 'MINMAX_SPEED','MINMAX_HEIGHT','Y']

    call = creat_features.total_call(data).reset_index(drop=True)
    total_run,mean_run = creat_features.total_mean_run(data)
    user_unsafe_speed_ratio, user_unsafe_trip_ratio = creat_features.unsafe_brake(data)
    user_unsafe_start_speed_ratio, user_unsafe_start_trip_ratio = creat_features.unsafe_start(data)

    data_others = pd.pivot_table(data, index=['TERMINALNO'],aggfunc=np.mean)
    terminalno = data[['TERMINALNO']].drop_duplicates().reset_index(drop=True)

    data_new = pd.concat([call, total_run,mean_run,user_unsafe_speed_ratio,user_unsafe_trip_ratio, \
                          user_unsafe_start_speed_ratio, user_unsafe_start_trip_ratio,\
                          data_others[columns_name[1:11]].reset_index(drop=True),\
                         terminalno], axis=1)

    if 'Y' in data.columns.tolist():  # 区分训练和测试集
        data_new['Y'] = data[['TERMINALNO','Y']].drop_duplicates(['TERMINALNO'])['Y'].reset_index(drop=True)#加上Y

    return data_new

def strategy1(data):
    y_user = data[data.Y != 0]
    noy_user = data[data.Y == 0]
    rand_arr = np.random.permutation(len(y_user))
    selected_noy_user = noy_user.loc[rand_arr]
    new_data = pd.concat(y_user, selected_noy_user)

    return new_data

def user_many_trip(data):
    def fun(x):
        print(x)
        print(x.index[0])  # trip start
        print(x.index[len(x.index) - 1])  # trip end

        res = data.iloc[x[len(x)-1]].LONGITUDE - data.iloc[x[0]].LONGITUDE
        longitude = math.abs(data.iloc[x.index[len(x.index) - 1]].LONGITUDE - data.iloc[x.index[0]].LONGITUDE)#每个行程中经度-差值
        latitude = math.abs(data.iloc[x.index[len(x.index) - 1]].LATITUDE - data.iloc[x.index[0]].LATITUDE)#每个行程中纬度-差值

        if longitude > 5 and latitude > 5:
            return 1
        else:
            return 0

    temp_data = data.groupby(['TERMINALNO', 'TRIP_ID']).transform(fun)
    # .drop_duplicates().index#每个行程ID的开始位置




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

def get_user_call_sum(data):
    call_sum_df = data[['TERMINALNO', 'TRIP_ID', 'CALLSTATE']].groupby(
        ['TERMINALNO', 'TRIP_ID', ]).sum()  # 每个用户在一个行程中打电话的次数
    user_trip = data[['TERMINALNO', 'TRIP_ID']].drop_duplicates()
    user_trip_call = pd.concat([user_trip.reset_index(drop=True), call_sum_df.reset_index(drop=True)],
                               axis=1)

    return user_trip_call

def preproess(train_df,test_df):
    """

    :param data: type-array
    :param label: type-array
    :return: data,label
    """
    columns_name = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED',
                    'CALLSTATE', 'Y']
    train_df = process_mistake_missing_duplicates(train_df)
    test_df = process_mistake_missing_duplicates(test_df)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    train_height = pd.DataFrame({'MINMAX_HEIGHT':scaler.fit_transform(train_df[['HEIGHT']])[:,0]})
    test_height = pd.DataFrame({'MINMAX_HEIGHT': scaler.transform(test_df[['HEIGHT']])[:, 0]})#test height

    scaler = MinMaxScaler()
    train_speed = pd.DataFrame({'MINMAX_SPEED': scaler.fit_transform(train_df[['SPEED']])[:, 0]})  # train height
    test_speed = pd.DataFrame({'MINMAX_SPEED': scaler.transform(test_df[['SPEED']])[:, 0]})  # test height

    train_new = pd.concat([train_height, train_speed, train_df[columns_name].reset_index(drop=True)], axis=1) # 修改好的数据拼接成原数据
    test_new = pd.concat([test_height, test_speed, test_df[columns_name[:9]].reset_index(drop=True) ], axis=1) # 修改好的数据拼接成原数据

    return train_new,test_new

def process():
    columns_name = ['TERMINALNO','TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    features_name = ['MINMAX_SPEED','MINMAX_HEIGHT','TOTAL_RUN','MEAN_RUN',\
                     'UNSAFE_SPEED_R','UNSAFE_TRIP_R','UNSAFE_START_SPEED_R','UNSAFE_START_TRIP_R']

    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)

    train_data,test_data = preproess(train_df,test_df)#预处理

    user_many_trip(train_data)

    train_data = strategy(train_data)#创造新特征
    test_data = strategy(test_data)

    from sklearn.tree import DecisionTreeRegressor
    estimator = DecisionTreeRegressor()
    estimator.fit(train_data[[features_name[1]] + features_name[4:8] + ['CALLSTATE']].values, train_data['Y'].values)
    predict_label = estimator.predict(test_data[[features_name[1]] + features_name[4:8] + ['CALLSTATE']].values)

    print_correlation(train_data,options=range(7,9))

    test_data['Pred'] = predict_label

    # 处理丢失的用户
    after_user_id = test_data['TERMINALNO'].drop_duplicates()
    before_user_id = test_df['TERMINALNO'].drop_duplicates()

    if len(before_user_id) > len(after_user_id):
        after_index = set(after_user_id.index.tolist())
        before_index = set(before_user_id.index.tolist())
        cha = before_index - after_index
        user_cha = test_df[['TERMINALNO']].iloc[list(cha)]
        user_cha['Pred'] = np.zeros((len(user_cha), 1))
        test_data = pd.concat([test_data[['TERMINALNO','Pred']], user_cha])

    write_result(test_data['TERMINALNO'].drop_duplicates(),test_data['Pred'])


if __name__ == "__main__":
    start = time.clock()
    process()
    elapsed = (time.clock() - start)
    print('whole time:' + str(elapsed))