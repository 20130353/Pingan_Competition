# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/5/8 10:07
# file: draw_class_num.py
# description:

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    list_0 = [3.99246 ,2.22276 ,5.86463 ,2.05231 ,4.40840 ,2.46930 ,3.96584 ,1.01164,8.66774 ]
    list_1 = [9.16641 ,2.05100 ,2.49167 ,4.40042 ,3.91738 ,5.61996 ,1.01393 ,2.19811 ,3.79620  ]
    n = 9
    x_label = 'Kmeans(k=9)'
    x_ticks = ['1','2','3','4','5','6','7','8','9']
    save_path = 'k=9.png'
    legend_label = ['y=0','y!=0']
    plt.figure(figsize=(9, 6))

    X = np.arange(n)  # X是1,2,3,4,5,6,7,8,柱的个数
    # uniform均匀分布的随机数，normal是正态分布的随机数，0.5-1均匀分布的数，一共有n个
    plt.bar(X, list_0, alpha=0.9, width=0.35, facecolor='lightskyblue', edgecolor='white', label='one', lw=1)
    plt.bar(X + 0.35, list_1, alpha=0.9, width=0.35, facecolor='yellowgreen', edgecolor='white', label='second', lw=1)
    plt.ylabel('data numbers(ten thousand)')
    plt.xlabel(x_label)

    plt.xticks(np.arange(n),x_ticks)
    plt.legend(legend_label,loc="best location")  # label的位置在左上，没有这句会找不到label去哪了
    plt.savefig(save_path)
    plt.show()