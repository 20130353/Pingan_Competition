# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/4/21 19:05
# file: creat_features.py
# description:
import numpy as np
import pandas as pd
import time
import math
from math import radians, cos, sin, asin, sqrt


def sub(c):
    return c - 1

def add(c):
    return c + 1


def trip_speed(data_df):#['TERMINALNO','TRIP_ID,''SPEED','LONGITUDE','LATITUDE','Y']
    """
    TTD:trip total distance
    STOPN:stop number
    STOPR:stop number/trip number
    'MAX_START_SPEED'
    'MAX_STOP_SPEED',
    'MAX_LAT': max LATITUDE
    'MAX_LON':max LONGITUDE
    'MAX_SPEED':max speed
    'MIN_SPEED':min speed
    'CALLSTARTN': call total times

    :param data_df:
    :return:
    """
    start = time.clock()

    data = []
    columns_name = ['TERMINALNO','TRIP_ID','TTD','STOPN','STOPR','MAX_START_SPEED'\
                        ,'MAX_STOP_SPEED','MAX_LAT','MAX_LON','MIN_LAT','MIN_LON','MAX_SPEED','MIN_SPEED','CALLSTARTN','RUN_RANGE']
    users = data_df.groupby(by=['TERMINALNO'])
    if 'Y' in data_df.columns.tolist():
        columns_name.append('Y')
        for key_user,user in users:
            trips = user.groupby(by=['TRIP_ID'])
            for key_trip,trip in trips:
                ttd = trip.MINMAX_SPEED.sum() * 50 / 3
                selected_trip = trip[trip.MINMAX_SPEED == 0]
                stops_index = selected_trip.index.tolist()
                stop = len(selected_trip)
                stop_rate = stop / len(trip)

                if trip.index.tolist()[0] in stops_index:
                    stops_index.remove(trip.index.tolist()[0])
                start_index = list(map(sub, stops_index))  # 数据位置
                if trip.index.tolist()[-1] in stops_index:
                    stops_index.remove(trip.index.tolist()[-1])
                end_index = list(map(add,stops_index))

                max_start_speed = trip.loc[start_index].MINMAX_SPEED.max()
                max_stop_speed = trip.loc[end_index].MINMAX_SPEED.max()

                max_lat = trip.LATITUDE.max()  # 最大经度
                max_lon = trip.LONGITUDE.max()  # 最大纬度
                min_lat = trip.LATITUDE.min()  # 最大经度
                min_lon = trip.LONGITUDE.min()  # 最大纬度
                run_range  = haversine(min_lon,min_lat,max_lon,max_lat)

                max_speed = trip.MINMAX_SPEED.max()  # 最大速度
                min_speed = trip.MINMAX_SPEED.min()  # 最小速度

                call_times = trip.CALLSTATE.sum()
                data.append([list(user.TERMINALNO)[0],list(trip.TRIP_ID)[0],ttd,stop,stop_rate\
                                ,max_start_speed,max_stop_speed,max_lat,max_lon,min_lat,min_lon,max_speed,min_speed\
                                ,call_times,run_range,list(trip.Y)[0]])
    else:
        for key_user, user in users:
            trips = user.groupby(by=['TRIP_ID'])
            for key_trip, trip in trips:
                ttd = trip.MINMAX_SPEED.sum() * 50 / 3
                stop = len(trip[trip.MINMAX_SPEED == 0])
                stop_rate = stop / len(trip)

                stops_index = trip[trip.MINMAX_SPEED == 0].index.tolist()
                if trip.index.tolist()[0] in stops_index:
                    stops_index.remove(trip.index.tolist()[0])
                start_index = list(map(sub, stops_index))  # 数据位置
                if trip.index.tolist()[-1] in stops_index:
                    stops_index.remove(trip.index.tolist()[-1])
                end_index = list(map(add, stops_index))

                max_start_speed = trip.loc[start_index].MINMAX_SPEED.max()
                max_stop_speed = trip.loc[end_index].MINMAX_SPEED.max()

                max_lat = trip.LATITUDE.max()  # 最大经度
                max_lon = trip.LONGITUDE.max()  # 最大纬度
                min_lat = trip.LATITUDE.min()  # 最大经度
                min_lon = trip.LONGITUDE.min()  # 最大纬度
                run_range = haversine(min_lon, min_lat, max_lon, max_lat)

                max_speed = trip.MINMAX_SPEED.max()  # 最大速度
                min_speed = trip.MINMAX_SPEED.min()  # 最小速度

                call_times = trip.CALLSTATE.sum()
                data.append([list(user.TERMINALNO)[0], list(trip.TRIP_ID)[0], ttd, stop, stop_rate \
                                , max_start_speed, max_stop_speed, max_lat, max_lon,min_lat,min_lon, max_speed, min_speed \
                                , call_times,run_range])

    new_data = pd.DataFrame(data,columns=columns_name)

    last = time.clock() - start
    print('trip_total_run time cost:' + str(last))
    return new_data


def long_time_driving(strategy_TIME):# ['TERMINALNO','TIME']
    '''
    按照用户的计算最长驾驶时间和是否长途驾驶
    :param strategy_TIME:
    :return: 时间间隔和是否疲劳驾驶
    '''
    start = time.clock() # 计时开始
    strategy_TIME = strategy_TIME.sort_values(by=['TERMINALNO', 'TIME'])
    users = strategy_TIME.groupby('TERMINALNO')
    ID = []
    time_space = []
    time_space_id = []
    for key_user, user in users:

        ID.append(key_user)
        user = user.sort_values(by='TIME')
        group_diff = user.diff()
        group_diff = group_diff.drop('TERMINALNO', axis=1)#计算当前组和上一组之间的差

        trip_num = 1
        trip_length = 0
        trip_length_MAX = 0
        for index, row in group_diff.iterrows():

            if (row['TIME'] <= 600):# 连续驾驶小于10分钟
                trip_length = trip_length + row['TIME']
            if (row['TIME'] >= 600):# 连续驾驶大于10分钟
                trip_num = trip_num + 1
                time_space_id.append(trip_length)
                if (trip_length > trip_length_MAX):
                    trip_length_MAX = trip_length
                trip_length = 0


    print('long drive time:' + str(time.clock()-start))
    return time_space

def long_time_driving1(data): #['TERMINALNO','TIME','Y']
    start = time.clock()  # 计时开始


    users = data.groupby(by=['TERMINALNO'])
    for key_user, user in users:
        user = user.sort_values(by='TIME').reset_index(drop=True)
        group_diff = user.diff()  # 计算当前组和上一组之间的差

        # 十分钟之内的时间间隔都是一次驾驶
        index = group_diff[(group_diff.TIME >= 600)].index.tolist()
        if 0 in index:
            index.remove(0)
        index = list(map(sub, index))  # 数据位置

        new_data = None
        trip_id = 1
        index.insert(0, 0)
        index.append(len(user))
        most_time = 0
        for i in range(1, len(index)):
            one_dis = user.loc[index[i - 1]:index[i] - 1]
            one_dis_time = len(one_dis) / 60.0
            one_dis['LONG_TIME_DRIVING'] = one_dis_time
            most_time = max(most_time,one_dis_time)
            if new_data is None:
                new_data = one_dis
            else:
                new_data = pd.concat([new_data, one_dis])

        new_data[new_data.TERMINALNO == user.TERMINALNO]['MOST_TIME'] = most_time
    print('long drive time1:' + str(time.clock() - start))
    return new_data


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    计算两个经纬度之间的距离
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def direction(data):# ['TERMINALNO','TRIP_ID','DIRECTION','SPEED','CALLSTATE']
    '''
    DIRDIFFERENCE:direction difference(方向差)
    :param data:
    :return: DIRDIFFERENCE
    '''
    start = time.clock()
    direction_array = data['DIRECTION'].values

    ###方向差量,夹角
    temp_difference = np.abs(direction_array[1:], direction_array[:-1])
    temp_difference[temp_difference > 180] = 360 - temp_difference[temp_difference > 180]
    temp_difference_value = np.zeros_like(direction_array)
    temp_difference_value[1:] = temp_difference

    ###方向变化量
    temp_direction_change = direction_array[1:] - direction_array[:-1]
    ###向右转
    temp_direction_change[temp_direction_change >= 0] = 1
    temp_direction_change[temp_direction_change <= -180] = 1
    ###向左转
    temp_direction_change[temp_direction_change < 0] = -1
    temp_direction_change[temp_direction_change >= 180] = -1
    temp_direction = np.zeros_like(direction_array)
    temp_direction[1:] = temp_direction_change

    ###每个行程的初始变化为0
    if 'Y' in data.columns.tolist():
        direction_df = pd.DataFrame(
            {'TERMINALNO': data['TERMINALNO'], 'TRIP_ID': data['TRIP_ID'], 'DIRDIFFERENCE': temp_difference_value,'Y':data['Y']})
    else:
        direction_df = pd.DataFrame(
            {'TERMINALNO': data['TERMINALNO'], 'TRIP_ID': data['TRIP_ID'], 'DIRDIFFERENCE': temp_difference_value})

    gd = direction_df.groupby([direction_df['TERMINALNO'], direction_df['TRIP_ID']])
    t = []
    for name, group in gd:
        t.append(group.index[0])
    direction_df['DIRDIFFERENCE'][t] = 0
    ###判断是否为急转弯
    direction_df['DIRDIFFERENCE'][direction_df['DIRDIFFERENCE'] < 120] = 0
    direction_df['DIRDIFFERENCE'][direction_df['DIRDIFFERENCE'] >= 120] = 1
    temp_speed = data['SPEED'].values
    temp_speed[temp_speed < 30] = 0
    temp_speed[temp_speed >= 30] = 1
    direction_df['DIRDIFFERENCE'] = direction_df['DIRDIFFERENCE'] * temp_speed

    return direction_df

    print('direction:' + str(time.clock()-start))