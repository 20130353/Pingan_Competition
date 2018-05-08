# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/5/4 12:18
# file: print_user.py
# description:

import pandas as pd
import time

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

def print_y(train_df):
    for i in range(30):
        max_v = train_df.Y.max()
        train_df = train_df[train_df.Y != max_v]
        print(str(i) + '_max_y:' + str(max_v))

def process():
    # 原始属性的名称
    columns_name = ['TERMINALNO','TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    train_df = pd.read_csv(path_train)
    print_y(train_df)

if __name__ == "__main__":
    start = time.clock()# 计时器
    process()
    elapsed = (time.clock() - start)
    print('whole time:' + str(elapsed))