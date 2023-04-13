import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib import rcParams

from scipy.interpolate import make_interp_spline

import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 14,
    "mathtext.fontset":'stix',
}
rcParams.update(config)


def iter_rayleign_true(p_init):
    G = np.array([[0.8,0.12],[0.15,1.2]])
    gamma = np.array([0.12,0.13])
    v = np.array([[0.25],[0.28]])
    O_bar =np.array([0.2,0.3])

    # G = np.array([[0.8, 0.12], [0.15, 1.2]])
    # gamma = np.array([0.2, 0.5])
    # v = np.array([[0.5], [0.8]])
    # O_bar = np.array([0.5, 0.5])

     # p_iter: [[0.05586284 0.10974677]] p_iter: [[0.9446592  1.04398486]]
    p_iter = copy.deepcopy(p_init)
    tol = 10e-8
    err = 1
    iter_num = 0
    max_iter_num = 15
    p1_iter_list_true = [p_init[0][0]]
    p2_iter_list_true = [p_init[0][1]]

    while err > tol or iter_num< max_iter_num:
        print("iter_num:", iter_num)
        p_temp = copy.deepcopy(p_iter)

        p_iter[0][0] = (v[0]+G[0][0] * p_iter[0][0]*np.log(1+gamma[0]*G[0][1]*p_iter[0][1]/(G[0][0]*p_iter[0][0])) )[0]/(G[0][0]*np.log(1/(1-O_bar[0])))
        p_iter[0][1] = (v[1] + G[1][1] * p_iter[0][1] * np.log(1 + gamma[1] * G[1][0] * p_iter[0][0] / (G[1][1] * p_iter[0][1])))[0] / (
                    G[1][1] * np.log(1 / (1 - O_bar[1])))
        err = max(np.abs(p_iter[0] - p_temp[0]))
        print("err:", err)
        p1_iter_list_true.append(p_iter[0][0])
        p2_iter_list_true.append(p_iter[0][1])
        # print("p_iter:", p_iter)
        # print("p_temp:", p_temp)
        # print("np.abs(p_iter - p_temp):", np.abs(p_iter - p_temp))
        iter_num = iter_num + 1
    print("p_iter:", p_iter)
    return p1_iter_list_true, p2_iter_list_true


def plaine_iteration(trained_beta1, trained_alpha1, trained_beta2, trained_alpha2, p_init):
    #p_iter: [[0.05586284 0.10974677]]
    p_iter = copy.deepcopy(p_init)
    p1_periter_list = []
    p2_periter_list = []
    tol = 10e-7
    err = 1
    iter_num = 0
    max_iter_num = 15
    p1_iter_list = [p_init[0][0]]
    p2_iter_list = [p_init[0][1]]

    while err > tol or iter_num< max_iter_num:
        print("iter_num:",iter_num)
        p_temp = copy.deepcopy(p_iter)
        for i in range(len(trained_alpha1)):
            p1_periter_list.append(np.dot(trained_beta1[i], p_iter.T)+trained_alpha1[i])
        for i in range(len(trained_alpha2)):
            p2_periter_list.append(np.dot(trained_beta2[i], p_iter.T) + trained_alpha2[i])
        p_iter[0][0] = min(p1_periter_list)
        p_iter[0][1] = min(p2_periter_list)
        err = max(np.abs(p_iter[0] - p_temp[0]))
        print("err:", err)
        # print("p_iter:", p_iter)
        # print("p_temp:", p_temp)
        # print("np.abs(p_iter - p_temp):", np.abs(p_iter - p_temp))
        iter_num = iter_num+1
        p1_iter_list.append(p_iter[0][0])
        p2_iter_list.append(p_iter[0][1])

    print("p_iter:", p_iter)
    return p1_iter_list, p2_iter_list


def cnls_iteration(coef_file_name, p_init):
    coef1 = pd.read_csv(coef_file_name)
    print("coef1.beta11:", coef1.beta11)

    trained_beta1 = np.array([coef1.beta11, coef1.beta12]).T
    trained_alpha1 = np.array([coef1.alpha1]).T

    trained_beta2 = np.array([coef1.beta21, coef1.beta22]).T
    trained_alpha2 = np.array([coef1.alpha2]).T

    p1_iter_list, p2_iter_list = plaine_iteration(trained_beta1, trained_alpha1, trained_beta2, trained_alpha2, p_init)
    return p1_iter_list, p2_iter_list


def lspa_iteration(coef_file_name1, coef_file_name2, p_init):
    coef1 = pd.read_csv(coef_file_name1)
    trained_beta1 = np.array([coef1.beta1, coef1.beta2]).T
    trained_alpha1 = np.array([coef1.alpha]).T

    coef2 = pd.read_csv(coef_file_name2)
    trained_beta2 = np.array([coef2.beta1, coef2.beta2]).T
    trained_alpha2 = np.array([coef2.alpha]).T

    plaine_iteration(trained_beta1, trained_alpha1, trained_beta2, trained_alpha2, p_init)


def myplot(p1_iter_list_true, p2_iter_list_true, p1_iter_list, p2_iter_list):
    x = np.linspace(0, len(p1_iter_list), len(p1_iter_list), endpoint=True)
    x_new = np.linspace(x.min(), x.max(), 300)
    y1_smooth = make_interp_spline(x, p1_iter_list_true)(x_new)
    y2_smooth = make_interp_spline(x, p2_iter_list_true)(x_new)
    y3_smooth = make_interp_spline(x, p1_iter_list)(x_new)
    y4_smooth = make_interp_spline(x, p2_iter_list)(x_new)
    # 设置图框的大小
    fig = plt.figure()

    # 绘图--阅读人数趋势
    plt.plot(y1_smooth,  # x轴数据
             linestyle='-',  # 折线类型
             linewidth=1.5,  # 折线宽度
             color='red',  # 折线颜色
             # marker='o',  # 点的形状
             markersize=4,  # 点的大小
             # markeredgecolor='black',  # 点的边框色
             markerfacecolor=None,
             alpha=1,
             label="$I_1(p)$")  # 添加标签

    plt.plot(y2_smooth,  # x轴数据
             linestyle='-.',  # 折线类型
             linewidth=1.5,  # 折线宽度
             color='red',  # 折线颜色
             # marker='o',  # 点的形状
             markersize=8,  # 点的大小
             # markeredgecolor='black',  # 点的边框色
             markerfacecolor=None,  # 点的填充色
             alpha=1,
             label="$I_2(p)$")  # 添加标签

    plt.plot(y3_smooth,  # x轴数据
             linestyle='-',  # 折线类型
             linewidth=1.5,  # 折线宽度
             color='blue',  # 折线颜色
             # marker='o',  # 点的形状
             markersize=5,  # 点的大小
             # markeredgecolor='black',  # 点的边框色
             markerfacecolor=None,  # 点的填充色
             alpha=1,
             label="$I_1(p)$")  # 添加标签

    plt.plot(y4_smooth,  # x轴数据
             linestyle='-.',  # 折线类型
             linewidth=1.5,  # 折线宽度
             color='blue',  # 折线颜色
             # marker='^',  # 点的形状
             markersize=6,  # 点的大小
             # markeredgecolor='black',  # 点的边框色
             markerfacecolor=None,  # 点的填充色
             alpha=1,
             label="$I_1(p)$")  # 添加标签


    # 添加标题和坐标轴标签
    # plt.title('公众号每天阅读人数和人次趋势图')
    plt.xlabel('Iteration step')
    plt.ylabel('Value')

    # 剔除图框上边界和右边界的刻度
    plt.tick_params(top='off', right='off')

    # 获取图的坐标信息
    # 用ax=plt.gca()获得axes对象
    ax = plt.gca()



    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()


if __name__ == '__main__':
    p_init = np.array([[5.5, 4.3]])
    p1_iter_list_true, p2_iter_list_true = iter_rayleign_true(p_init)
    coef_file_name = 'cnls_coef_raylein.csv'
    p_init = np.array([[3.5,2.5]])
    p1_iter_list, p2_iter_list = cnls_iteration(coef_file_name, p_init)

    myplot(p1_iter_list_true, p2_iter_list_true, p1_iter_list, p2_iter_list)


    # p_init = np.array([[8.5,8.5]])
    # plaine_num = 18
    # lspa_iteration('lspa_coef_rayleign_I1_n'+str(plaine_num)+'.csv', 'lspa_coef_rayleign_I2_n'+str(plaine_num)+'.csv', p_init)

