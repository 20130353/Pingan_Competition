# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/4/21 19:05
# file: creat_features.py
# description:
import numpy as np
import pandas as pd

def user_run(data_df):
    '''
    计算用户的总行程
    :param data_df:
    :return:
    '''
    total_run = 50/3 * data_df[['TERMINALNO','SPEED']].pivot_table(index='TERMINALNO',aggfunc=np.sum)
    total_run.rename(columns = {'SPEED':'TOTAL_RUN'},inplace=True)
    total_run = total_run.reset_index(drop=True)

    return total_run

def user_call(data_df):
    '''
    计算用户的总打电话次数
    :param data_df:
    :return:
    '''
    # call = pd.pivot_table(data_df[['TERMINALNO', 'CALLSTATE']], index=['TERMINALNO'], aggfunc=np.sum)
    call = data_df[['TERMINALNO', 'CALLSTATE']].groupby(['TERMINALNO']).count()
    call = call.reset_index(drop=True)
    return call

def user_trip_call(data):
    '''
    计算用户在每个行程中的打电话次数
    :param data:
    :return:
    '''
    call_sum_df = data[['TERMINALNO', 'TRIP_ID', 'CALLSTATE']].groupby(
        ['TERMINALNO', 'TRIP_ID', ]).sum()  # 每个用户在一个行程中打电话的次数
    user_trip = data[['TERMINALNO', 'TRIP_ID']].drop_duplicates()
    user_trip_call = pd.concat([user_trip.reset_index(drop=True), call_sum_df.reset_index(drop=True)],
                               axis=1)

    return user_trip_call

def unsafe_brake(data_df):
    '''
    计算用户的不安全熄火比例
    :param data_df:
    :return:
    '''

    index = data_df[(data_df.SPEED == 0)].index.tolist()

    def sub(c):
        return c - 1

    if 0 in index:
        index.remove(0)
    index = list(map(sub, index))

    stops = data_df.iloc[index]
    dangers = stops[stops.SPEED > 30] #熄灭速度>30

    user_dangers =  dangers[['TERMINALNO', 'SPEED']].groupby(['TERMINALNO']).count()
    user_unsafe_speed_ratio = (user_dangers/data_df[['TERMINALNO','SPEED']].groupby(['TERMINALNO']).count()).fillna(0)#占所有开车的比例

    temp_df = data_df[['TERMINALNO', 'TRIP_ID']].groupby(['TERMINALNO']).count()
    temp_df.rename(columns={'TRIP_ID':'SPEED'},inplace=True)
    user_unsafe_trip_ratio = (user_dangers/temp_df).fillna(0)#占所有行程的比例

    user_unsafe_speed_ratio.rename(columns={'SPEED': 'UNSAFE_SPEED_R'}, inplace=True)
    user_unsafe_trip_ratio.rename(columns={'SPEED': 'UNSAFE_TRIP_R'}, inplace=True)

    user_unsafe_speed_ratio = user_unsafe_speed_ratio.reset_index(drop=True)
    user_unsafe_trip_ratio = user_unsafe_trip_ratio.reset_index(drop=True)

    return user_unsafe_speed_ratio,user_unsafe_trip_ratio

def unsafe_start(data_df):
    '''
    计算用户的不安全启动比例
    :param data_df:
    :return:
    '''
    index = data_df[(data_df.SPEED == 0)].index.tolist()

    def sub(c):
        return c + 1

    if len(data_df) in index:
        index.remove(len(data_df))
    index = list(map(sub, index))

    stops = data_df.iloc[index]
    dangers = stops[stops.SPEED > 20] #启动速度>30

    user_dangers =  dangers[['TERMINALNO', 'SPEED']].groupby(['TERMINALNO']).count()
    user_unsafe_speed_ratio = (user_dangers/data_df[['TERMINALNO','SPEED']].groupby(['TERMINALNO']).count()).fillna(0)#占所有开车的比例

    temp_df = data_df[['TERMINALNO', 'TRIP_ID']].groupby(['TERMINALNO']).count()
    temp_df.rename(columns={'TRIP_ID':'SPEED'},inplace=True)
    user_unsafe_trip_ratio = (user_dangers/temp_df).fillna(0)#占所有行程的比例

    user_unsafe_speed_ratio.rename(columns={'SPEED': 'UNSAFE_START_SPEED_R'}, inplace=True)
    user_unsafe_trip_ratio.rename(columns={'SPEED': 'UNSAFE_START_TRIP_R'}, inplace=True)

    user_unsafe_speed_ratio = user_unsafe_speed_ratio.reset_index(drop=True)
    user_unsafe_trip_ratio = user_unsafe_trip_ratio.reset_index(drop=True)

    return user_unsafe_speed_ratio,user_unsafe_trip_ratio