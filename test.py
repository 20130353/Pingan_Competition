# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/5/7 21:15
# file: test.py
# description:
import numpy as np
import time
import pandas as pd

def pringf(c):
    data = np.array(c) + 1
    print(len(c))
    print(c)
x = [[1,2,3,4],[3,4,5,4],[5,6,5,4]]
df = pd.DataFrame(x,columns=['a','b','c','d'])
data = df.apply(pringf,axis=1)
print()
