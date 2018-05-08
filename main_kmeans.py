# -*- coding:utf8 -*-

# this file uses multi cross input

import os
import csv
import pandas as pd
import numpy as np
import time

import copy

from sklearn.cluster import KMeans

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

def kmeans(data):
    # 设定不同k值以运算
    for k in range(5,6):
        print('---------------' + str(k) + '-------------------')
        clf = KMeans(n_clusters=k)  # KMeans算法
        s = clf.fit(data)  # 加载数据集合
        centroids = clf.labels_
        # print('------1.each class num')
        # for i in range(k):
        #     print(str(i) + ':' + str(sum(centroids == i)))

        print('------2.each class samples')
        for i in range(k):
            cen_data = data[centroids == i].reset_index(drop=True)
            rand_arr = np.random.permutation(len(cen_data))
        #
        #     print('---' + str(i) + 'class samples')
        #     if len(cen_data) > 5:
        #         for j in range(5):
        #             print(cen_data.loc[rand_arr[j]])
        #     else:
        #         for j in range(len(cen_data)):
        #             print(cen_data.loc[rand_arr[j]])
        #
        # print('------3.each class centrids')
        # centroids = clf.cluster_centers_
        # for i in range(k):
        #     print('---' + str(i) + ' sample:' + str(centroids[i]))

def process():
    # 原始属性的名称
    columns_name = ['TERMINALNO','TIME','TRIP_ID','LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE','Y']
    # 新创造特征的名称
    features_name = ['MINMAX_SPEED','MINMAX_HEIGHT','TTD','STOPN','STOPR']

    train_df = pd.read_csv(path_train)

    noy_df = train_df[train_df.Y == 0]
    print('Y=0')
    kmeans(noy_df)

    # y_df = train_df[train_df.Y != 0]
    # print('Y>0')
    # kmeans(y_df)

if __name__ == "__main__":
    # start = time.clock()# 计时器
    process()
    # elapsed = (time.clock() - start)
    # print('whole time:' + str(elapsed))