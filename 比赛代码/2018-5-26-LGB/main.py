# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import time
import datetime
from math import radians, cos, sin, asin, sqrt
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# path
path_train = "/data/dm/train.csv"  # 训练文件路径
path_test = "/data/dm/test.csv"  # 测试文件路径
path_result_out = "model/result.csv" #预测结果文件路径

def create_rand(c):
    global seed
    return random.uniform(0,seed)

def process_y0(data):
    data0 = data[data.Pred==0]
    global seed
    seed = data[data.Pred!=0].min()
    y = data0[['Pred']].apply(create_rand, axis=1)
    data0.Pred = y
    new_data = pd.concat([data0,data[data.Pred!=0]])
    return new_data

def split_users(data, iteration):
    '''
    使Y=0和Y!=0的样本相等，最后一次使用剩下的全部样本
    :param data: 数据
    :param iteration: 第几次迭代
    :return:
    '''
    y_user = data[data.target !=0].reset_index(drop=True)
    noy_user = data[data.target == 0].reset_index(drop=True)

    if iteration != 5: # 最后一次
        selected_noy_user = noy_user.loc[iteration*len(y_user): (iteration+1)*len(y_user)]
    else:
        selected_noy_user = noy_user.loc[iteration*len(y_user):len(noy_user)]

    new_data = pd.concat([y_user,selected_noy_user])

    return new_data

def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
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

def preprocess(data_df):
    mean_value = data_df['SPEED'][data_df.SPEED != -1].mean()  # 去掉缺失值之后的均值
    data_df.SPEED = data_df['SPEED'].replace([-1], [mean_value])  # 均值速度填充缺失值

    data_df.CALLSTATE = data_df['CALLSTATE'].replace([-1, 2, 3, 4],
                                                     [0, 1, 1, 0])  # 修改callstate的状态为-1,2,3,4-没打电话，打电话，打电话和没打电话
    data_df = data_df[(73 < data_df.LONGITUDE) & (data_df.LONGITUDE < 135) \
                      & (18 < data_df.LATITUDE) & (data_df.LATITUDE < 53) \
                      & (data_df.DIRECTION != -1) \
                      & (-150 < data_df.HEIGHT) & (data_df.HEIGHT < 6700)]
    # 中国经度73~135之间, # 中国维度20~53之间, # 删除方向丢失的数据,# 中国高度20~53之间
    if 'Y' in data_df.columns.tolist():
        data_df.loc[(data_df.Y >= 300),'Y'] = 148
    return data_df

def process():
    start_all = datetime.datetime.now()
    # read train data
    data = pd.read_csv(path_train)
    data = preprocess(data)
    train1 = []
    alluser = data['TERMINALNO'].nunique()
    # Feature Engineer, 对每一个用户生成特征:
    # trip特征, record特征(数量,state等)

    # 地理位置特征(location,海拔,经纬度等), 时间特征(星期,小时等), 驾驶行为特征(速度统计特征等)
    for item in data['TERMINALNO'].unique():
        # print('user NO:',item)
        temp = data.loc[data['TERMINALNO'] == item, :]
        temp.index = range(len(temp))
        # trip 特征
        num_of_trips = temp['TRIP_ID'].nunique()
        # record 特征
        num_of_records = temp.shape[0]  # trip数据总条数
        num_of_state = temp[['TERMINALNO', 'CALLSTATE']]

        nsh = num_of_state.shape[0]  # 每种电话状态所占比例
        num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE'] == 0].shape[0] / float(nsh)
        num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE'] == 1].shape[0] / float(nsh)
        del num_of_state

        ###总路程
        total_run = 50 / 3 * sum(temp['SPEED'].values)

        ###停顿的比例
        num_of_speed_0 = temp.loc[(temp.SPEED == 0)].shape[0] / num_of_trips

        ###用户急刹的比例
        speed = temp[['TERMINALNO','SPEED']]
        index = np.array(speed.loc[speed.SPEED == 0].index.tolist()) - 1 ## 急刹车的位置
        temp_speed = speed.loc[index]
        brakes = temp_speed.loc[temp_speed.SPEED > 10].shape[0] / num_of_trips

        ###用户快速启动的比例
        index =  np.array(speed.loc[speed.SPEED == 0].index.tolist()) + 1
        temp_speed = speed.loc[index]
        start = temp_speed.loc[temp_speed.SPEED > 10].shape[0] / num_of_trips
        del speed

        ###急转弯的比例
        direction_diff = temp[['DIRECTION']].diff().astype('float32')
        shape_turn = direction_diff.loc[((np.abs(direction_diff.DIRECTION) >= 90) & (temp.SPEED > 10))].shape[0] / num_of_trips

        ### 转弯时打电话的比例
        call_turn = direction_diff.loc[((np.abs(direction_diff.DIRECTION) >= 90) & (temp.CALLSTATE  == 1))].shape[0] / num_of_trips

        ###高速驾驶时打电话的比例
        call_speed = temp.loc[(temp.SPEED > 10) & (temp.CALLSTATE == 1)].shape[0] / num_of_trips

        ###上坡的比例
        height_diff = temp[['HEIGHT']].diff().astype('float32')
        uphill = height_diff.loc[height_diff.HEIGHT > 1].shape[0] / num_of_trips
        downhill = height_diff.loc[height_diff.HEIGHT < 1].shape[0] / num_of_trips

        ###下坡速度很快的比例
        down_speed = height_diff.loc[(height_diff.HEIGHT < 1) & (temp.SPEED > 20)].shape[0] / num_of_trips

        ### 上下坡打电话的比例
        up_down_call = height_diff.loc[(abs(height_diff.HEIGHT) > 1) & (temp.CALLSTATE == 1)].shape[0] / num_of_trips

        ### 上下坡转弯的比例
        up_down_turn = direction_diff.loc[((np.abs(direction_diff.DIRECTION) >= 90) & (abs(direction_diff.DIRECTION) > 1))].shape[0] / num_of_trips

        ### 经纬度特征
        startlong = temp.loc[0, 'LONGITUDE']
        startlat = temp.loc[0, 'LATITUDE']
        hdis1 = haversine1(startlong, startlat, 113.9177317, 22.54334333)  # 距离广东的距离

        ### 海拔特征
        mean_height = temp['HEIGHT'].mean()
        var_height = temp['HEIGHT'].var()
        # height_gap = mean_height - 43.5 # 和北京市海拔高度差值

        # 时间特征
        temp['weekday'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).weekday())
        temp['hour'] = temp['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x).hour)
        hour_state = np.zeros([24, 1])
        for i in range(24):
            hour_state[i] = temp.loc[temp['hour'] == i].shape[0] / float(nsh)  # 0-24中每个小时所占比例

        # 夜间驾驶
        night_driving = (hour_state[0] + hour_state[1] + hour_state[2] + hour_state[3] + hour_state[4] + hour_state[5])[0]

        # 驾驶行为特征
        mean_speed = temp['SPEED'].mean()
        var_speed = temp['SPEED'].var()

        #周一-周日每天开车的频率
        weekday = temp.loc[(temp['weekday'] >= 1) & (temp['weekday'] <= 5)].shape[0] / num_of_records
        weekend = temp.loc[(temp['weekday'] >= 6) & (temp['weekday'] <= 7)].shape[0] / num_of_records

        # 添加label
        target = temp.loc[0, 'Y']

        # 所有特征
        feature = [item, num_of_trips, num_of_records, num_of_state_0, num_of_state_1,\
                   mean_speed, var_speed, mean_height,var_height, weekday,weekend,night_driving
            ,total_run,num_of_speed_0,brakes,start,shape_turn,call_turn
            ,call_speed,uphill,downhill,down_speed,up_down_call,up_down_turn
            , float(hour_state[0]), float(hour_state[1]), float(hour_state[2]), float(hour_state[3]),
                   float(hour_state[4]), float(hour_state[5])
            , float(hour_state[6]), float(hour_state[7]), float(hour_state[8]), float(hour_state[9]),
                   float(hour_state[10]), float(hour_state[11])
            , float(hour_state[12]), float(hour_state[13]), float(hour_state[14]), float(hour_state[15]),
                   float(hour_state[16]), float(hour_state[17])
            , float(hour_state[18]), float(hour_state[19]), float(hour_state[20])
            , float(hour_state[21]),float(hour_state[22]), float(hour_state[23])
            , hdis1
            , target]
        train1.append(feature)
    train1 = pd.DataFrame(train1)

    # 特征命名
    featurename = ['item', 'num_of_trips', 'num_of_records', 'num_of_state_0', 'num_of_state_1',\
                   'mean_speed', 'var_speed', 'mean_height','var_height','weekday','weekend','night_driving'\
        ,'total_run','num_of_speed_0','brakes','start','shape_turn','call_turn','call_speed'
        ,'uphill','downhill','down_speed','up_down_call','up_down_turn'
        , 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11'
        , 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18'\
        , 'h19', 'h20','h21','h22','h23'
        , 'dis'
        , 'target']
    train1.columns = featurename

    # print("train data process time:", (datetime.datetime.now() - start_all).seconds)
    # 特征使用
    feature_use = ['item', 'num_of_trips', 'num_of_records', 'num_of_state_0', 'num_of_state_1',\
                   'mean_speed', 'var_speed', 'mean_height','var_height','weekday','weekend','night_driving'
        , 'total_run', 'num_of_speed_0','brakes','start','shape_turn','call_turn','call_speed'
        ,'uphill','downhill','down_speed','up_down_call','up_down_turn'
        , 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11'
        , 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18'
        , 'h19', 'h20','h21','h22','h23'
        , 'dis'
        ]

    # The same process for the test set
    data = pd.read_csv(path_test)
    data = preprocess(data)
    test1 = []
    for item in data['TERMINALNO'].unique():
        # print('user NO:',item)
        temp = data.loc[data['TERMINALNO'] == item, :]
        temp.index = range(len(temp))
        # trip 特征
        num_of_trips = temp['TRIP_ID'].nunique()
        # record 特征
        num_of_records = temp.shape[0]
        num_of_state = temp[['TERMINALNO', 'CALLSTATE']]
        nsh = num_of_state.shape[0]
        num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE'] == 0].shape[0] / float(nsh)
        num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE'] == 1].shape[0] / float(nsh)
        del num_of_state

        ###总路程
        total_run = 50 / 3 * sum(temp['SPEED'].values)

        ###停顿的比例
        num_of_speed_0 = temp.loc[(temp.SPEED == 0)].shape[0] / num_of_trips

        ###用户急刹的比例
        speed = temp[['TERMINALNO', 'SPEED']]
        index = np.array(speed.loc[speed.SPEED == 0].index.tolist()) - 1  ## 急刹车的位置
        temp_speed = speed.loc[index]
        brakes = temp_speed.loc[temp_speed.SPEED > 10].shape[0] / num_of_trips

        ###用户快速启动的比例
        index = np.array(speed.loc[speed.SPEED == 0].index.tolist()) + 1
        temp_speed = speed.loc[index]
        start = temp_speed.loc[temp_speed.SPEED > 10].shape[0] / num_of_trips
        del speed

        ###急转弯的比例
        direction_diff = temp[['DIRECTION']].diff().astype('float32')
        shape_turn = direction_diff.loc[((np.abs(direction_diff.DIRECTION) >= 90) & (temp.SPEED > 10))].shape[
                         0] / num_of_trips

        ### 转弯时打电话的比例
        call_turn = direction_diff.loc[((np.abs(direction_diff.DIRECTION) >= 90) & (temp.CALLSTATE == 1))].shape[
                        0] / num_of_trips

        ###高速驾驶时打电话的比例
        call_speed = temp.loc[(temp.SPEED > 10) & (temp.CALLSTATE == 1)].shape[0] / num_of_trips

        ###上坡的比例
        height_diff = temp[['HEIGHT']].diff().astype('float32')
        uphill = height_diff.loc[height_diff.HEIGHT > 1].shape[0] / num_of_trips
        downhill = height_diff.loc[height_diff.HEIGHT < 1].shape[0] / num_of_trips

        ###下坡速度很快的比例
        down_speed = height_diff.loc[(height_diff.HEIGHT < 1) & (temp.SPEED > 20)].shape[0] / num_of_trips

        ### 上下坡打电话的比例
        up_down_call = height_diff.loc[(abs(height_diff.HEIGHT) > 1) & (temp.CALLSTATE == 1)].shape[0] / num_of_trips

        ### 上下坡转弯的比例
        up_down_turn = direction_diff.loc[((np.abs(direction_diff.DIRECTION) >= 90)\
                                        & (abs(direction_diff.DIRECTION) > 1))].shape[0] / num_of_trips

        ### 经纬度特征
        startlong = temp.loc[0, 'LONGITUDE']
        startlat = temp.loc[0, 'LATITUDE']
        hdis1 = haversine1(startlong, startlat, 113.9177317, 22.54334333)  # 距离广东的距离

        ### 海拔特征
        mean_height = temp['HEIGHT'].mean()
        var_height = temp['HEIGHT'].var()
        # height_gap = mean_height - 43.5  # 和北京市海拔高度差值

        # 时间特征
        temp['weekday'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).weekday())

        # 周一-周日每天开车的频率
        weekday = temp.loc[(temp['weekday'] >= 1) & (temp['weekday'] <= 5)].shape[0] / num_of_records
        weekend = temp.loc[(temp['weekday'] >= 6) & (temp['weekday'] <= 7)].shape[0] / num_of_records

        temp['hour'] = temp['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x).hour)
        hour_state = np.zeros([24, 1])
        for i in range(24):
            hour_state[i] = temp.loc[temp['hour'] == i].shape[0] / float(nsh)

        # 夜间驾驶
        night_driving = (hour_state[0] + hour_state[1] + hour_state[2] + hour_state[3] + hour_state[4] + hour_state[5])[0]

        # 驾驶行为特征
        mean_speed = temp['SPEED'].mean()
        var_speed = temp['SPEED'].var()

        # test标签设为-1
        target = -1.0
        feature = [item, num_of_trips, num_of_records, num_of_state_0, num_of_state_1,\
                   mean_speed, var_speed, mean_height,var_height,weekday,weekend,night_driving \
            , total_run, num_of_speed_0,brakes,start,shape_turn,call_turn
            , call_speed,uphill,downhill,down_speed,up_down_call,up_down_turn
            , float(hour_state[0]), float(hour_state[1]), float(hour_state[2]), float(hour_state[3]),
                   float(hour_state[4]), float(hour_state[5])
            , float(hour_state[6]), float(hour_state[7]), float(hour_state[8]), float(hour_state[9]),
                   float(hour_state[10]), float(hour_state[11])
            , float(hour_state[12]), float(hour_state[13]), float(hour_state[14]), float(hour_state[15]),
                   float(hour_state[16]), float(hour_state[17])
            , float(hour_state[18]),float(hour_state[19]), float(hour_state[20])\
            , float(hour_state[21]),float(hour_state[22]), float(hour_state[23])
            , hdis1
            , target]
        test1.append(feature)
    # make predictions for test data
    test1 = pd.DataFrame(test1)
    test1.columns = featurename

    # 采用lgb回归预测模型，具体参数设置如下
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.01, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    # 训练、预测
    model_lgb.fit(train1[feature_use], train1['target'])
    final_res = pd.DataFrame({'Id':test1['item'].drop_duplicates(),'Pred':model_lgb.predict(test1[feature_use])})
    final_res = process_y0(final_res)  # 处理结果为0的y值
    final_res.to_csv(path_result_out, header=True, index=False)

    ###LGB 模型的特征重要性
    importances = model_lgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    print('---------LGB feature importance----------')
    print(
        sorted(zip(map(lambda x: round(x, 4), model_lgb.feature_importances_), feature_use),
               reverse=True))

    # ###随机森林特征的重要性
    # model = RandomForestRegressor()
    # model.fit(train1[feature_use], train1['target'])
    # importances = model.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # print('---------RandomForestRegressor feature importance----------')
    # print(
    #     sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), feature_use),
    #            reverse=True))
    #
    # ###特征递归消除
    # lr = LinearRegression()
    # # rank all features, i.e continue the elimination until the last one
    # rfe = RFE(lr, n_features_to_select=1)
    # rfe.fit(train1[feature_use], train1['target'])
    # importances = rfe.ranking_
    # indices = np.argsort(importances)[::-1]
    # print('---------LinearRegression feature importance----------')
    # print(
    # sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), feature_use),
    #        reverse=True))

if __name__ == "__main__":
    start = time.clock()# 计时器
    warnings.filterwarnings("ignore")
    process()
    print('whole_time:' +  str(round(time.clock() - start,2)))
