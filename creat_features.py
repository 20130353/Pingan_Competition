import numpy as np
import pandas as pd
import time
from math import radians, cos, sin, asin, sqrt

def sub(c):
    return c - 1

def add(c):
    return c + 1

def trip_lon_lat(data_df):#['TERMINALNO','TRIP_ID,''SPEED','LONGITUDE','LATITUDE','Y']
    """

    'MAX_LAT': max LATITUDE
    'MAX_LON':max LONGITUDE
    'MIN_LAT': min LATITUDE
    'MIN_LON':min LONGITUDE
    'MEAN_LON':mean LONGITUDE
    'MEAN_LAT': mean LATITUDE
    'RUN_RANGE' : max run range
    'CALLSTARTN': call total times

    :param data_df:
    :return:
    """
    start = time.clock()

    group_data = data_df[['TERMINALNO','TRIP_ID','LONGITUDE']].groupby(by=['TERMINALNO','TRIP_ID']) # 维度
    max_lon = group_data.max().LONGITUDE  # 最大纬度
    min_lon = group_data.min().LONGITUDE  # 最大纬度
    mean_lon = group_data.mean().LONGITUDE  # 最大纬度

    group_data = data_df[['TERMINALNO', 'TRIP_ID', 'LATITUDE']].groupby(by=['TERMINALNO', 'TRIP_ID'])# 经度
    max_lat = group_data.max().LATITUDE  # 最大纬度
    min_lat = group_data.min().LATITUDE  # 最大纬度
    mean_lat = group_data.mean().LATITUDE  # 最大纬度

    new_data = pd.DataFrame({'MAX_LON': max_lon, 'MIN_LON': min_lon, 'MAX_LAT': max_lat \
                                , 'MIN_LAT': min_lat, 'MEAN_LON': mean_lon, 'MEAN_LAT': mean_lat})

    run_range = new_data[['MAX_LON','MAX_LAT','MIN_LON','MIN_LAT']].apply(haversine,axis=1)# 最大行驶范围
    new_data['RUN_RANGE'] = run_range
    new_data['CALL_TIMES'] = data_df[['TERMINALNO', 'TRIP_ID', 'CALLSTATE']].sum().CALLSTATE # 打电话总次数

    print('lat_lon:' +  str(round(time.clock() - start,2)))
    return new_data

def trip_speed(data): #['TERMINALNO','TRIP_ID','SPEED']
    '''
    # 'MAX_START_SPEED': 最大启动速度
    # 'MAX_STOP_SPEED': 最大熄灭速度
    # 'MIN_START_SPEED'：最小启动速度
    # 'MIN_STOP_SPEED'：最大熄灭速度
    TTD: 行驶总路程
    # STOPN: 停车次数
    # STOPR: 停车比例
    MAX_SPEED：最大速度
    MIN_SPEED： 最小速度
    MEAN_SPEED: 平均速度
    :param data:
    :return:
    '''
    start = time.clock()
    # stop_speed_index = data[data.SPEED == 0].index.tolist()
    #
    # #start----------------------- 启动速度 和 熄灭速度-- 有问题
    # previous_index = list(np.array(stop_speed_index)-1)
    # if -1 in previous_index:
    #     previous_index.remove(-1)
    #
    # previous_speed = data.loc[previous_index]
    # trip_spped = previous_speed.groupby(by=['TERMINALNO', 'TRIP_ID'])
    # max_start_speed = trip_spped.max()
    # min_start_speed = trip_spped.min()
    #
    # after_index = list(np.array(stop_speed_index) + 1)
    # last_index = len(data)
    # if last_index in after_index:
    #     after_index.remove(last_index)
    #
    # after_speed = data.loc[after_index]
    # trip_speed = after_speed.groupby(by=['TERMINALNO', 'TRIP_ID'])
    # max_stop_speed = trip_speed.max()
    # min_stop_speed = trip_speed.min()
    #
    # new_data = pd.concat([max_start_speed, max_stop_speed, min_start_speed \
    #                          , min_stop_speed], axis=1)
    # new_data.columns = ['MAX_START_SPEED', 'MAX_STOP_SPEED' \
    #     , 'MIN_START_SPEED', 'MIN_STOP_SPEED']
    # # end----------------------- 启动速度 和 熄灭速度

    new_data = pd.DataFrame()
    new_data['TTD'] = data.groupby(by=['TERMINALNO', 'TRIP_ID']).sum().SPEED.values * 50 / 3  # 行驶路程
    # new_data['STOP_N'] = data[data.SPEED == 0].groupby(by=['TERMINALNO', 'TRIP_ID']).count().SPEED # 停车次数--有问题

    group_data = data.groupby(by=['TERMINALNO', 'TRIP_ID'])
    # new_data['STOPR'] = new_data['STOP_N'].values / float(np.array(group_data.count().SPPED))  # 停车比例--有问题
    new_data['MAX_SPEED'] = list(group_data.min().SPEED) # 最大速度
    new_data['MIN_SPEED'] = list(group_data.max().SPEED) # 最小速度
    new_data['MEAN_SPEED'] = list(group_data.mean().SPEED)  # 最小速度

    print('trip_speed:' +  str(round(time.clock() - start,2)))
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

    print('long_driving:' +  str(round(time.clock() - start,2)))
    return time_space

# 这段待修改
# def long_time_driving1(data): #['TERMINALNO','TIME','Y']
#     start = time.clock()  # 计时开始
#
#     users = data.groupby(by=['TERMINALNO'])
#     for key_user, user in users:
#         user = user.sort_values(by='TIME').reset_index(drop=True)
#         group_diff = user.diff()  # 计算当前组和上一组之间的差
#
#         # 十分钟之内的时间间隔都是一次驾驶
#         index = group_diff[(group_diff.TIME >= 600)].index.tolist()
#         if 0 in index:
#             index.remove(0)
#         index = list(map(sub, index))  # 数据位置
#
#         new_data = None
#         trip_id = 1
#         index.insert(0, 0)
#         index.append(len(user))
#         most_time = 0
#         for i in range(1, len(index)):
#             one_dis = user.loc[index[i - 1]:index[i] - 1]
#             one_dis_time = len(one_dis) / 60.0
#             one_dis['LONG_TIME_DRIVING'] = one_dis_time
#             most_time = max(most_time,one_dis_time)
#             if new_data is None:
#                 new_data = one_dis
#             else:
#                 new_data = pd.concat([new_data, one_dis])
#
#         new_data[new_data.TERMINALNO == user.TERMINALNO]['MOST_TIME'] = most_time
#     return new_data


def haversine(data):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    计算两个经纬度之间的距离
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度

    lon1, lat1, lon2, lat2 = map(radians, list(data))

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def difference(datavector):
    temp_difference = datavector[1:] - datavector[:-1]
    difference = np.zeros_like(datavector)
    difference[1:] = temp_difference
    return difference

def direction(data):# ['TERMINALNO','TRIP_ID','DIRECTION','SPEED','CALLSTATE','HEIGHT','Y']
    '''
    DIR_DIFFERENCE:方向差
    'SLOPE':坡度
    'SPEED_DIFFERENCE':加速度
    'HEIGHT_DIFFERENCE':高度差
    'CALL_LEFT':打电话左转
    'CALL_RIGHT':打电话右转

    :param data:
    :return:
    '''
    start = time.clock()
    direction_array = data['DIRECTION'].values

    ###速度
    spped_array = data['SPEED'].values
    spped_array[spped_array == -1] = spped_array.mean()
    temp_speed_difference = difference(spped_array)

    ###高度
    height_array = data['HEIGHT'].values
    temp_height_difference = difference(height_array)

    ###方向
    direction_array = data['DIRECTION'].values
    ###去掉缺失值-1
    miss_value = np.where(direction_array == -1)
    direction_array[miss_value] = (direction_array[miss_value[0] - 1] + direction_array[miss_value[0] + 1]) / 2
    # print(direction_array)

    ###方向变化量
    temp_direction_change = direction_array[1:] - direction_array[:-1]
    ###向右为正
    temp = [temp_direction_change <= -180]
    temp_direction_change[temp] = temp_direction_change[temp] + 360
    ###向左为负
    temp = [temp_direction_change >= 180]
    temp_direction_change[temp] = temp_direction_change[temp] - 360
    temp_direction = np.zeros_like(direction_array)
    temp_direction[1:] = temp_direction_change
    # 方向差值
    temp_difference_value = np.abs(temp_direction)

    ###每个行程的初始变化为0
    direction_df = pd.DataFrame({'TERMINALNO': data['TERMINALNO'].values, 'TRIP_ID': data['TRIP_ID'].values})
    gd = direction_df.groupby([direction_df['TERMINALNO'], direction_df['TRIP_ID']])
    t = []
    for name, group in gd:
        t.append(group.index[0])
    ###loc可以改变原值
    # print(direction_df['DIRDIFFERENCE'][t])
    # print('====')
    temp_difference_value[t] = 0
    temp_direction[t] = 0
    temp_speed_difference[t] = 0
    temp_height_difference[t] = 0

    ###判断是否为急转弯
    temp1 = temp_difference_value < 90
    temp2 = temp_difference_value >= 90
    temp_difference_value[temp1] = 0
    temp_difference_value[temp2] = 1

    temp = np.zeros_like(spped_array)
    temp[spped_array >= 15] = 1
    temp_difference_value = temp_difference_value * temp

    turn_left = temp_direction < -30
    turn_right = temp_direction > 30

    ###电话分为0和1，未知和呼入，呼出，连接称为1，断连为0
    call_array = data['CALLSTATE'].values
    weight_callstate = spped_array * call_array

    ###打电话时左转
    call_left = call_array * turn_left
    ###打电话时右转
    call_right = call_array * turn_right

    ###路程
    journey = (spped_array) * 16.67
    ###高度差
    highdiff = temp_height_difference
    ###角度,坡度
    temp = [journey != 0]
    slope = np.zeros_like(highdiff)
    slope[temp] = np.arctan(highdiff[temp] / journey[temp])

    data_df = pd.DataFrame({'TERMINALNO': direction_df['TERMINALNO'], 'TRIP_ID': direction_df['TRIP_ID']\
                            ,'DIR_DIFFERENCE': temp_difference_value, 'SPEED_DIFFERENCE': np.abs(temp_speed_difference)\
                            ,'HEIGHT_DIFFERENCE': np.abs(temp_height_difference)\
                            ,'CALL_LEFT': call_left, 'CALL_RIGHT': call_right, 'SLOPE': slope})

    if 'Y' in data.columns.tolist():
        data_df['Y'] = data['Y']

    new_data = data_df.groupby(by=['TERMINALNO','TRIP_ID']).mean().reset_index()
    print('direction:' +  str(round(time.clock() - start,2)))
    return new_data



def timestamp_datetime(data): #['TERMINALNO','TRIP_ID','TIME']
    '''
    'TIRED_DRIVING': 是否是疲劳驾驶
    'FRE_DRIVING': 最经常驾驶的时间段
    :param data:
    :return:
    '''
    start = time.clock()

    def ab(c):  # 返回数据的第三个
        return c[3]

    #获取所有数据的效识数据
    data_times = data['TIME'].map(time.localtime)
    hours = list(map(ab, list(data_times))) # 获取驾驶的时间- 在某个小时
    data['DRIVING_HOURS'] = hours
    data['TIREDT_DRIVING'] = hours # 是否是疲劳驾驶

    data1 = data[(data.TIREDT_DRIVING >= 0) & (data.TIREDT_DRIVING <= 3)]
    data2 = data[(data.TIREDT_DRIVING >= 18) & (data.TIREDT_DRIVING <= 20)]
    data3 = data[(data.TIREDT_DRIVING >= 3) & (data.TIREDT_DRIVING <= 18)]
    data4 = data[(data.TIREDT_DRIVING >= 20) & (data.TIREDT_DRIVING <= 23)]
    new_data1 = pd.concat([data1,data2])
    new_data2 = pd.concat([data3,data4])
    new_data1.TIREDT_DRIVING=1
    new_data2.TIREDT_DRIVING=2
    new_data = pd.concat([new_data1,new_data2])

    final_data = new_data[['TERMINALNO','TRIP_ID','TIREDT_DRIVING']].groupby(by=['TERMINALNO','TRIP_ID']).max() #是否疲劳驾驶取最大
    final_data1 = new_data[['TERMINALNO', 'TRIP_ID', 'DRIVING_HOURS']].groupby(by=['TERMINALNO', 'TRIP_ID']).agg(
        lambda x: np.mean(pd.Series.mode(x)))# 最经常驾驶时间

    final_new_data = pd.DataFrame({'TIREDT_DRIVING':final_data.TIREDT_DRIVING, 'DRIVING_HOURS':final_data1.DRIVING_HOURS})

    print('time_features:' +  str(round(time.clock() - start,2)))

    return final_new_data
