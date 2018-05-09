# -*- coding:utf8 -*-

# this file uses multi cross input

import os
import csv
import pandas as pd
import numpy as np
import time

import copy

import creat_features as CF
import common_tool as CT
from preproess import preproess_fun

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def split_users(data, iteration):
    '''
    使Y=0和Y!=0的样本相等，最后一次使用剩下的全部样本
    :param data: 数据
    :param iteration: 第几次迭代
    :return:
    '''
    y_user = data[data.Y !=0].reset_index(drop=True)
    noy_user = data[data.Y == 0].reset_index(drop=True)

    if iteration != 5: # 最后一次
        selected_noy_user = noy_user.loc[iteration*len(y_user): (iteration+1)*len(y_user)]
    else:
        selected_noy_user = noy_user.loc[iteration*len(y_user):len(noy_user)]

    new_data = pd.concat([y_user,selected_noy_user])

    return new_data

def strategy(data):
    '''
    统一创造新特征，然后并入汇总成新特征
    :param data:
    :return:
    '''

    data = data.sort_values(by=['TERMINALNO','TRIP_ID']).reset_index(drop=True)

    time_data = CF.timestamp_datetime(data[['TERMINALNO','TRIP_ID','TIME']]) #'TIREDT_DRIVING'，'DRIVING_HOURS'
    lat_lon_data = CF.trip_lon_lat(data[['TERMINALNO','TRIP_ID','LONGITUDE','LATITUDE','CALLSTATE']])
    #'MAX_LON','MAX_LAT','MIN_LON','MIN_LAT'，'MEAN_LON'，'MEAN_LAT'，RUN_RANGE，CALL_TIMES
    trip_speed_data = CF.trip_speed(data[['TERMINALNO', 'TRIP_ID','SPEED']]) #TTD,MAX_SPEED,MIN_SPEED,MEAN_SPEED

    if 'Y' in data.columns.tolist():
        direction_data = CF.direction(data[['TERMINALNO', 'TRIP_ID', 'DIRECTION', 'SPEED', 'CALLSTATE','HEIGHT', 'Y']])
        #DIR_DIFFERENCE, 'SLOPE','SPEED_DIFFERENCE','HEIGHT_DIFFERENCE','CALL_LEFT','CALL_RIGHT'

    else:
        direction_data = CF.direction(data[['TERMINALNO', 'TRIP_ID', 'DIRECTION', 'SPEED', 'CALLSTATE','HEIGHT']])

    new_data1 = pd.concat([direction_data,trip_speed_data],axis=1).reset_index(drop=True).reset_index(drop=True)
    new_data2 = pd.concat([time_data,lat_lon_data],axis=1).reset_index(drop=True)
    new_data = pd.concat([new_data1,new_data2],axis=1)

    # CT.evaluate_feature('TTD', new_data)# 评价TTD特征的相关性
    # CT.evaluate_feature('STOPN', new_data)  # 评价TTD特征的相关性
    # CT.evaluate_feature('STOPR', new_data)  # 评价TTD特征的相关性

    return new_data

def process():
    # 原始属性的名称
    columns_name = ['TERMINALNO','TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    # 新创造特征的名称
    features_name = ['TIREDT_DRIVING','DRIVING_HOURS','MAX_LON','MAX_LAT','MIN_LON','MIN_LAT'\
            ,'MEAN_LON','MEAN_LAT','RUN_RANGE','CALL_TIMES','DIR_DIFFERENCE', 'SLOPE'\
            ,'SPEED_DIFFERENCE','HEIGHT_DIFFERENCE','CALL_LEFT','CALL_RIGHT']

    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)

    origin_train_data,origin_test_data = preproess_fun(train_df,test_df)#预处理

    # 6:1 训练分类器
    final_res = None
    origin_train_data = strategy(origin_train_data)
    origin_test_data = strategy(origin_test_data)
    test_data = origin_test_data[['TERMINALNO']]
    for iteration in range(6):# Y=0-6:1-Y!=0
        train_data = split_users(origin_train_data, iteration=iteration)#分割训练样本

        from sklearn.tree import DecisionTreeRegressor# 决策树分类器
        estimator = DecisionTreeRegressor()
        estimator.fit(train_data[features_name].values, train_data['Y'].values)
        predict_label = estimator.predict(origin_test_data[features_name].values)

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

        test_data = test_data.sort_values(axis = 0,ascending = True,by = 'TERMINALNO')

        if final_res is None:
            final_res = copy.deepcopy(test_data[['TERMINALNO','Pred']])
        else:
            final_res['Pred' + str(iteration)] = test_data['Pred']

    final_max_res = final_res[['TERMINALNO','Pred']].groupby('TERMINALNO').max()
    final_max_res = CT.process_y0(final_max_res)
    CT.write_result(list(final_res['TERMINALNO'].drop_duplicates().values), final_max_res['Pred'].values)

    # 不交叉预测结果
    # origin_train_data = strategy(origin_train_data)
    # origin_test_data = strategy(origin_test_data)
    # test_data = origin_test_data[['TERMINALNO']]
    # from sklearn.tree import DecisionTreeRegressor  # 决策树分类器
    # estimator = DecisionTreeRegressor()
    # estimator.fit(origin_train_data[features_name[2:]].values, origin_train_data['Y'].values)
    # predict_label = estimator.predict(origin_test_data[features_name[2:]].values)
    # test_data['Pred'] = predict_label
    # final_max_res = test_data[['TERMINALNO','Pred']].groupby('TERMINALNO').max()
    # CT.write_result(list(test_data['TERMINALNO'].drop_duplicates().values), final_max_res['Pred'].values)

if __name__ == "__main__":
    start = time.clock()# 计时器
    process()
    elapsed = (time.clock() - start)
    print('whole time:' + str(elapsed))