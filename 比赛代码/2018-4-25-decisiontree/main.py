# -*- coding:utf8 -*-

# this file uses multi cross input

import os
import csv
import pandas as pd
import numpy as np
import time

import copy

import creat_features
from common_tool import *
from preproess import preproess_fun

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def strategy_1(data,iteration):
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

    columns_name = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED',
                    'CALLSTATE', 'MINMAX_SPEED','MINMAX_HEIGHT','Y']

    call = creat_features.user_call(data)
    user_run= creat_features.user_run(data)
    user_unsafe_speed_ratio, user_unsafe_trip_ratio = creat_features.unsafe_brake(data)
    user_unsafe_start_speed_ratio, user_unsafe_start_trip_ratio = creat_features.unsafe_start(data)

    data_others = data.groupby(['TERMINALNO']).mean()
    # data_others = pd.pivot_table(data, index=['TERMINALNO'],aggfunc=np.mean)
    terminalno = data[['TERMINALNO']].drop_duplicates().reset_index(drop=True)

    data_new = pd.concat([call, user_run,user_unsafe_speed_ratio,user_unsafe_trip_ratio, \
                          user_unsafe_start_speed_ratio, user_unsafe_start_trip_ratio,\
                          data_others[columns_name[3:11] + [columns_name[1]]].reset_index(drop=True),\
                         terminalno], axis=1)

    if 'Y' in data.columns.tolist():  # 区分训练和测试集
        data_new['Y'] = data[['TERMINALNO','Y']].drop_duplicates(['TERMINALNO'])['Y'].reset_index(drop=True)#加上Y

    return data_new

def process():
    # 原始属性的名称
    columns_name = ['TERMINALNO','TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    # 新创造特征的名称
    features_name = ['MINMAX_SPEED','MINMAX_HEIGHT','USER_RUN',\
                     'UNSAFE_SPEED_R','UNSAFE_TRIP_R','UNSAFE_START_SPEED_R','UNSAFE_START_TRIP_R']

    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)

    origin_train_data,origin_test_data = preproess_fun(train_df,test_df)#预处理

    final_res = None
    origin_train_data = strategy(origin_train_data)
    origin_test_data = strategy(origin_test_data)
    test_data = origin_test_data[['TERMINALNO']]
    for iteration in range(6):# Y=0-6:1-Y!=0
        train_data = strategy_1(origin_train_data,iteration=iteration)#分割训练样本

        from sklearn.tree import DecisionTreeRegressor# 决策树分类器
        estimator = DecisionTreeRegressor()
        estimator.fit(train_data[[columns_name[8],features_name[0]]].values, train_data['Y'].values)
        predict_label = estimator.predict(origin_test_data[[columns_name[8],features_name[0]]].values)

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

    write_result(list(final_res['TERMINALNO'].drop_duplicates().values),final_res[['Pred']].mean(1))


if __name__ == "__main__":
    start = time.clock()# 计时器
    process()
    elapsed = (time.clock() - start)
    print('whole time:' + str(elapsed))