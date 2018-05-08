# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/5/7 21:15
# file: test.py
# description:
import numpy as np
import time
x = range(100000000000)


def add(c):
    return c ** 2

start = time.clock()
d1 = map(add, x)
print(str(time.clock()-start))

start = time.clock()
d2 = [c ** 2 for c in x]
print(str(time.clock()-start))

start = time.clock()
d3 = [add(c) for c in x]
print(str(time.clock()-start))