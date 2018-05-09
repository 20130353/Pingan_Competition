import pandas as pd
import time

def adjust_user_order(data_df):
    #['TERMINALNO', 'TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    '''
    根据用户驾驶时间调整trip_id
    如果用户两次trip的时间差至少是5分钟
    :return:
    '''
    start = time.clock() # 计时开始
    data_df = data_df.sort_values(by='TIME').reset_index(drop=True)
    data_df['TIME_DIFF'] = data_df[['TIME']].diff()

    gd = data_df.groupby(by=['TERMINALNO','TRIP_ID'])
    t = []
    for name, group in gd:
        t.append(group.index[0]) 
    data_df.loc[t].TIME_DIFF = 0 # 每个用户的第一条数的时间差是0

    index = data_df[(data_df.TIME_DIFF >= 300)].index.tolist()
    index.insert(0,0)
    index.append(len(index))

    for i in range(1,len(index)):
        data_df.loc[index[i-1]:index[i]-1].TRIP_ID = i

    print('adjust_user_order:' + str(round(time.clock() - start,2)))
    return data_df

def adjust_y(data_df):
    # ['TERMINALNO', 'TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    '''
    :param data_df:
    :return:
    '''
    start = time.clock()
    if data_df.Y.max() < 300:
        return data_df

    new_data =data_df[data_df.Y >= 300]
    new_data.Y = 148
    data_df = data_df[data_df.Y < 300]
    for i in range(9):
        maxv = data_df.Y.max()
        selected_data = data_df[data_df.Y == maxv]
        selected_data.Y = maxv - 10
        new_data = pd.concat([new_data,selected_data])
        data_df = data_df[data_df.Y != maxv]

    print('adjust_y:' +  str(round(time.clock() - start,2)))
    return new_data

def process_mistake_missing_duplicates(data_df):
    #['TERMINALNO', 'TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    """
    去除数据中的缺失值，错误值和重复值（删除时间后，去掉所有重复的）
    :param data_df:
    :return:
    """
    start = time.clock()
    columns_name = ['TERMINALNO', 'TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    mean_value = data_df['SPEED'][data_df.SPEED != -1].mean()#去掉缺失值之后的均值
    speed_df = pd.DataFrame(data_df['SPEED'].replace([-1], [mean_value]))  # 均值速度填充缺失值

    call_df = pd.DataFrame(data_df['CALLSTATE'].replace([-1, 2, 3, 4], [0, 1, 1, 0]))  # 修改callstate的状态为-1,2,3,4-没打电话，打电话，打电话和没打电话
    data_df = data_df[(73 < data_df.LONGITUDE) & (135 > data_df.LONGITUDE)] # 中国经度73~135之间
    data_df = data_df[(18 < data_df.LATITUDE) & (53 > data_df.LATITUDE)]  # 中国维度20~53之间
    data_df = data_df[data_df.DIRECTION != -1] # 删除方向丢失的数据
    data_df = data_df[(-150 < data_df.HEIGHT) & (6700 > data_df.HEIGHT)]  # 中国高度20~53之间

    if 'Y' in data_df.columns.tolist():
        no_repeat_data = data_df.drop_duplicates(columns_name.remove('TIME'))  # 除去time之后的重复值
    else:
        columns_name.remove('TIME')
        columns_name.remove('Y')
        no_repeat_data = data_df.drop_duplicates(columns_name)  # 去除重复数据

    print('missing_value:' +  str(round(time.clock() - start,2)))
    return no_repeat_data

def preproess_fun(train_df,test_df):
    """
    :param data: type-array
    :param label: type-array
    :return: data,label
    """
    start = time.clock()
    columns_name = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED',
                    'CALLSTATE', 'Y']
    # train_df = adjust_user_order(train_df)#内存溢出
    # test_df = adjust_user_order(test_df)

    train_df = process_mistake_missing_duplicates(train_df)
    test_df = process_mistake_missing_duplicates(test_df)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler() # 归一化高度
    train_height = pd.DataFrame({'MINMAX_HEIGHT':scaler.fit_transform(train_df[['HEIGHT']])[:,0]})# train height
    test_height = pd.DataFrame({'MINMAX_HEIGHT': scaler.transform(test_df[['HEIGHT']])[:, 0]})#test height

    scaler = MinMaxScaler()# 归一化速度
    train_speed = pd.DataFrame({'MINMAX_SPEED': scaler.fit_transform(train_df[['SPEED']])[:, 0]})  # train height
    test_speed = pd.DataFrame({'MINMAX_SPEED': scaler.transform(test_df[['SPEED']])[:, 0]})  # test height

    train_df = adjust_y(train_df)

    train_new = pd.concat([train_height, train_speed, train_df[columns_name].reset_index(drop=True)], axis=1) # 修改好的数据拼接成原数据
    test_new = pd.concat([test_height, test_speed, test_df[columns_name[:9]].reset_index(drop=True) ], axis=1) # 修改好的数据拼接成原数据

    print('preproess_fun:' +  str(round(time.clock() - start,2)))
    return train_new,test_new

