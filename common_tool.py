# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/4/24 9:23
# file: common_tool.py
# description:

import pandas as pd

def write_result(id,pre_lable):
    """

    :param id:type-series
    :param pre_lable:type-array
    :return:nothing
    """
    dataframe = pd.DataFrame({'Id': id, 'Pred': pre_lable}, dtype=float)
    dataframe = pd.pivot_table(dataframe, index=['Id'])
    dataframe.to_csv("model/test.csv", index=True, sep=',')

def evaluate_feature(feature_name,data):
    index = list(data.columns).index(feature_name)
    train_data = data.drop(feature_name,axis=1)
    from sklearn.linear_model import LinearRegression
    regr = LinearRegression().fit(train_data,data[feature_name].values)
    print(feature_name + '_cor:' + str(regr.coef_[index]))

def cluster(data):
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=3).fit(data[data.Y==0].values)
    # print(kmeans.cluster_centers_)
    # for i,each in enumerate(kmeans.labels_):

    return 0
