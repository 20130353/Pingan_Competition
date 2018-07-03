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