import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler

def adjust_user_order(data_df):
    #['TERMINALNO', 'TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    '''
    根据用户驾驶时间调整trip_id
    如果用户两次trip的时间差至少是5分钟
    :return:
    '''
    start = time.clock()  # 计时开始
    data_df['TIME_DIFF'] = data_df[['TIME']].diff()

    first_trip_index = data_df.groupby(by=['TERMINALNO']).apply(lambda x: x.index[0]).sort_values()
    data_df.loc[first_trip_index, 'TIME_DIFF'] = 0  # 设置每个用户的第一条数据的时间差是0

    index = data_df[(data_df.TIME_DIFF >= 300) | (data_df.TIME_DIFF == 0)].index.tolist()
    index.append(data_df.index.tolist()[-1])

    for i in range(1, len(index)):
        data_df.loc[index[i - 1]:index[i] - 1, 'TRIP_ID'] = i
    data_df.loc[index[i], 'TRIP_ID'] = i  # 最后一个trip

    print('adjust_user_order:' + str(round(time.clock() - start, 2)))
    return data_df['TRIP_ID']

# def preproess_fun(data,**param):
#     """
#     :param data: type-array
#     :param label: type-array
#     :return: data,label
#     """
#     start = time.clock()
#
#     mean_value = data['SPEED'][data.SPEED != -1].mean()  # 去掉缺失值之后的均值
#     data.SPEED = data['SPEED'].replace([-1], [mean_value])  # 均值速度填充缺失值
#
#     # 修改callstate的状态为-1,2,3,4-没打电话，打电话，打电话和没打电话
#     data.CALLSTATE = data['CALLSTATE'].replace([-1, 2, 3, 4], [0, 1, 1, 0])
#
#     # 中国经度73~135之间, # 中国维度20~53之间, # 删除方向丢失的数据,# 中国高度20~53之间
#     data = data[(73 < data.LONGITUDE) & (data.LONGITUDE < 135) \
#                 & (18 < data.LATITUDE) & (data.LATITUDE < 53) \
#                 & (data.DIRECTION != -1) \
#                 & (-150 < data.HEIGHT) & (data.HEIGHT < 6700)]
#
#     data['MINMAX_HEIGHT'] = MinMaxScaler().fit_transform(data[['HEIGHT']])[:, 0]  # test height
#     data['MINMAX_SPEED'] = MinMaxScaler().fit_transform(data[['SPEED']])[:, 0]  # test height
#
#     print('preproess_fun:' +  str(round(time.clock() - start,2)))
#     return data

