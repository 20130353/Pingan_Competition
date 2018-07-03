# -*- coding:utf8 -*-

# this file uses multi cross input

import warnings
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from math import radians, cos, sin, asin, sqrt
import common_tool as CT

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
path_result_out = "model/result.csv"# 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式
columns_name = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION'\
        ,'HEIGHT', 'SPEED', 'CALLSTATE','Y']

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

def haversine(data):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    计算两个经纬度之间的距离,将十进制度数转化为弧度
    """
    lon1, lat1, lon2, lat2 = map(radians, list(data))
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def difference(datavector):
    difference[1:] = datavector[1:] - datavector[:-1]
    difference[0] = 0
    return difference

def strategy(data):
    ###转换数据类型----------------------------------------------------
    data1 = data[['TERMINALNO', 'TIME']].astype('int64')
    data1.columns = ['ID','TIME']
    data2 =  data[['CALLSTATE']].astype('int8')
    data3 = data[[ 'TRIP_ID']].astype('int16')

    if 'Y' in data.columns.tolist():
        c = ['LONGITUDE', 'LATITUDE', 'DIRECTION','HEIGHT', 'SPEED','Y']
    else:
        c = ['LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED']
    data4 = data[c].astype('float16')

    data = pd.concat([data1,data2,data3,data4],axis=1)
    del data1,data2,data3,data4
    # print(data.info(memory_usage='deep'))
    print('1')

    ### 数据预处理----------------------------------------------------
    mean_value = data['SPEED'][data.SPEED != -1].mean()  # 去掉缺失值之后的均值
    data.SPEED = data['SPEED'].replace([-1], [mean_value])  # 均值速度填充缺失值

    # 修改callstate的状态为-1,2,3,4-没打电话，打电话，打电话和没打电话
    data.CALLSTATE = data['CALLSTATE'].replace([-1, 2, 3, 4], [0, 1, 1, 0]).astype('int8')

    # 中国经度73~135之间, # 中国维度20~53之间, # 删除方向丢失的数据,# 中国高度20~53之间
    data = data[(73 < data.LONGITUDE) & (data.LONGITUDE < 135) \
                & (18 < data.LATITUDE) & (data.LATITUDE < 53) \
                & (data.DIRECTION != -1) \
                & (-150 < data.HEIGHT) & (data.HEIGHT < 6700)]

    data['MINMAX_HEIGHT'] = MinMaxScaler().fit_transform(data[['HEIGHT']])[:, 0].astype('float16')  # test height
    data['MINMAX_SPEED'] = MinMaxScaler().fit_transform(data[['SPEED']])[:, 0].astype('float16')  # test height
    data = data.sort_values(by=['ID', 'TRIP_ID', 'TIME']).reset_index(drop=True)
    # print(data.info(memory_usage='deep'))
    # print('2')

    ###创造新特征----------------------------------------------------
    hours = data['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x).hour)  # 获取驾驶的时间- 在某个小时
    data  =pd.concat([data,pd.DataFrame({'DRIVING_HOURS':hours,\
                                         'TIRED_DRIVING':np.zeros(data.shape[0])}).astype('int8')],axis=1)
    # print(data.info(memory_usage='deep'))
    # print('3')
    data.loc[((data.TIRED_DRIVING >= 0) & (data.TIRED_DRIVING <= 3) | \
              (data.TIRED_DRIVING >= 18) & (data.TIRED_DRIVING <= 20)), 'TIRED_DRIVING'] = 1  # 是疲劳驾驶

    # 计算速度，高度，方向的才差值
    data1_diff = data[['SPEED','HEIGHT']].diff().astype('float16')
    data1_diff.columns = ['SPEED_DIF','HEIGHT_DIF']
    data2_diff = data[['DIRECTION']].diff().astype('float32')
    data2_diff.columns = ['DIR_DIF']
    data = pd.concat([data,data1_diff,data2_diff],axis=1)
    data.loc[(data.DIR_DIF <= -180), 'DIR_DIF'] = \
        data.loc[(data.DIR_DIF < -180), 'DIR_DIF'] + 360
    data.loc[(data.DIR_DIF >= 180), 'DIR_DIF'] = \
        data.loc[(data.DIR_DIF >= 180), 'DIR_DIF'] - 360
    trip_index = data.groupby(by=['ID', 'TRIP_ID']).apply(lambda x: x.index[0]).sort_values()
    temp = data
    data.loc[trip_index,'SPEED_DIF'] = 0
    data.loc[trip_index,'HEIGHT_DIF'] = 0
    data.loc[trip_index,'DIR_DIF'] = 0
    # print(data.info(memory_usage='deep'))
    # print('6')

    #转弯时速度
    data['SHAPE_TURN'] = np.zeros(data.shape[0]).astype('int8')  # 转弯时速度
    data.loc[((np.abs(data.DIR_DIF) >= 90) & (data.SPEED > 15)), 'SHAPE_TURN'] = 1
    # print(data.info(memory_usage='deep'))
    # print('7')

    #打电话时速度
    data['CALL_LEFT'] = data['CALLSTATE']
    data['CALL_RIGHT'] =np.zeros(data.shape[0]).astype('int8')
    data.loc[data.DIR_DIF >= -30, 'CALL_LEFT'] = 0
    data.loc[data.DIR_DIF <= 30, 'CALL_RIGHT'] = 0
    # print(data.info(memory_usage='deep'))
    # print('9')

    #坡度
    data['SLOPE'] = data['HEIGHT_DIF'].astype('float32')
    data.loc[(data.SPEED == 0), 'SLOPE'] = 0
    data.loc[(data.SPEED != 0), 'SLOPE'] = data['SLOPE'] / (data['SPEED'] * 16.67)
    # print(data.info(memory_usage='deep'))
    # print('10')

    #用户ID & Trip_ID
    group_data = data.groupby(by=['ID', 'TRIP_ID'])
    max_group_data = group_data.max()
    min_group_data = group_data.min()
    mean_group_data = group_data.mean()
    sum_group_data = group_data.sum()
    var_group_data = group_data.var()
    # print('11')

    #高度，方向，坡度需要float32，其他的需要float16

    max_data = max_group_data[['TIRED_DRIVING','LONGITUDE','LATITUDE','SPEED'\
        ,'SPEED_DIF','HEIGHT_DIF','DIR_DIF','SLOPE','SHAPE_TURN','CALL_LEFT','CALL_RIGHT']]
    max_data.columns = ['MAX_TIRED_DRIVING','MAX_LON','MAX_LAT','MAX_SPEED'\
        ,'MAX_SPEED_DIF','MAX_HEIGHT_DIF','MAX_DIR_DIF','MAX_SLOPE','MAX_SHAPE_TURN','MAX_CALL_LEFT','MAX_CALL_RIGHT']

    min_data = min_group_data[['TIRED_DRIVING', 'LONGITUDE', 'LATITUDE', 'SPEED' \
        , 'SPEED_DIF', 'HEIGHT_DIF', 'DIR_DIF', 'SLOPE']]
    min_data.columns = ['MIN_TIRED_DRIVING', 'MIN_LON', 'MIN_LAT', 'MIN_SPEED' \
        , 'MIN_SPEED_DIF', 'MIN_HEIGHT_DIF', 'MIN_DIR_DIF', 'MIN_SLOPE']

    mean_data = mean_group_data[['TIRED_DRIVING', 'LONGITUDE', 'LATITUDE', 'SPEED' \
        , 'SPEED_DIF', 'HEIGHT_DIF', 'DIR_DIF', 'SLOPE']]
    mean_data.columns = ['MEAN_TIRED_DRIVING', 'MEAN_LON', 'MEAN_LAT', 'MEAN_SPEED' \
        , 'MEAN_SPEED_DIF', 'MEAN_HEIGHT_DIF', 'MEAN_DIR_DIF', 'MEAN_SLOPE']

    sum_data = sum_group_data[['CALLSTATE','SHAPE_TURN','CALL_LEFT','CALL_RIGHT','TIRED_DRIVING']].astype('int8')
    sum_data.columns = ['SUM_CALL','SUM_SHAPE_TURN','SUM_CALL_LEFT','SUM_CALL_RIGHT','SUM_TIRED_DRIVING']

    var_data = var_group_data[['CALLSTATE', 'SHAPE_TURN', 'CALL_LEFT', 'CALL_RIGHT', 'TIRED_DRIVING']]
    var_data.columns = ['VAR_CALL', 'VAR_SHAPE_TURN', 'VAR_CALL_LEFT', 'VAR_CALL_RIGHT', 'VAR_TIRED_DRIVING']

    new_data = pd.concat([max_data,min_data,mean_data,sum_data,var_data],axis=1)
    del max_data,min_data,mean_data,sum_data
    print(new_data.info(memory_usage='deep'))
    # print('12')

    new_data['TTD'] = sum_group_data.SPEED.astype('float32') * 50 / 3  # 行驶路程
    new_data['DRIVING_HOURS'] = data[['ID', 'TRIP_ID', 'TIRED_DRIVING']] \
        .groupby(by=['ID', 'TRIP_ID']).agg(lambda x: np.mean(pd.Series.mode(x)))  # 最经常驾驶时间
    # print(new_data.info(memory_usage='deep'))
    print('13')

    #可以改成和某一点之间的距离
    run_range = new_data[['MAX_LON', 'MAX_LAT', 'MIN_LON', 'MIN_LAT']].apply(haversine, axis=1)  # 对每行使用haversine函数，计算最大行驶范围
    new_data['RUN_RANGE'] = run_range.astype('float32')
    # print(new_data.info(memory_usage='deep'))
    print('14')

    user_trip = list(group_data.indices.keys())  # 用户ID
    new_data['ID'] = list(map(lambda x: x[1], user_trip))
    if 'Y' in data.columns.tolist():
        new_data['Y'] = mean_group_data.Y
    # print(new_data.info(memory_usage='deep'))
    CT.print_na(new_data)
    print('15')
    return new_data

def process():
    print('train')
    train_df = pd.read_csv(path_train)
    origin_train_data = strategy(train_df)
    origin_train_data.fillna(0)#防止结果出现na
    del  train_df

    #test
    # print('test')
    # test_df = pd.read_csv(path_test)
    # origin_test_data = strategy(test_df)
    # origin_test_data.fillna(0)#防止结果出现na
    # del test_df
    #
    # # 6:1 训练分类器
    # print('classifier')
    # test_data = origin_test_data[['ID']]
    # final_res = test_data[['ID']]
    # estimator = DecisionTreeRegressor()# 决策树分类器
    # for iteration in range(6):# Y=0-6:1-Y!=0
    #     train_data = split_users(origin_train_data, iteration=iteration)#分割训练样本
    #     estimator.fit(train_data.drop('Y',axis=1), train_data['Y'])
    #     predict_label = estimator.predict(origin_test_data.values)
    #     if iteration == 0:
    #         final_res['Pred'] = predict_label
    #     else:
    #         final_res['Pred'] = final_res['Pred'].values + predict_label
    # final_res['Pred'] = final_res.Pred.values / 6
    # final_max_res = final_res[['ID','Pred']].groupby('ID').max()
    # final_max_res = CT.process_y0(final_max_res) # 处理结果为0的y值
    # result = final_max_res.rename(columns={'item': 'Id', 'Pred': 'Pred'})
    # result.to_csv(path_result_out, header=True, index=False)

if __name__ == "__main__":
    start = time.clock()# 计时器
    warnings.filterwarnings("ignore")
    process()
    print('whole_time:' +  str(round(time.clock() - start,2)))