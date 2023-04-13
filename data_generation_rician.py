import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data_ricean():
    data_num = 1000
    G = np.array([[0.8,0.12],[0.15,1.2]])
    gamma = np.array([0.2,0.5])
    v = np.array([[0.5],[0.8]])
    O_bar =np.array([0.5,0.5])
    # G = np.array([[0.8, 0.12], [0.15, 1.2]])
    # gamma = np.array([0.2, 0.3])
    # v = np.array([[0.25], [0.28]])
    # O_bar = np.array([0.2, 0.3])

    p = np.abs(10 * np.random.randn(2, data_num))
    I1 = []
    I2 = []
    sample_num = 2000
    chisquare_df = 0.5

    for i in range(data_num):
        print("i:", i)
        good_sample1_num = 0
        good_sample2_num = 0

        for j in range(sample_num):
            h11 = np.random.chisquare(df=chisquare_df, size=1)
            h12 = np.random.chisquare(df=chisquare_df, size=1)
            h21 = np.random.chisquare(df=chisquare_df, size=1)
            h22 = np.random.chisquare(df=chisquare_df, size=1)
            t1 = gamma[0]*(G[0][1]*h12*p[:,i][1]+v[0])/(G[0][0]*h11)
            if (t1<=p[:,i][0]):
                good_sample1_num += 1
            # print("t1:", t1[0])
            # print("sample_list1:", sample_list1)
            t2 = gamma[1] * (G[1][0] * h21*p[:,i][0] + v[1]) / (G[1][1] * h22)
            if (t2<=p[:,i][1]):
                good_sample2_num += 1

        # print("good_sample1_num:", good_sample1_num)
        # print("good_sample2_num:", good_sample2_num)

        tI1 = -p[:,i][0]*np.log(good_sample1_num/sample_num)
        tI2 = -p[:,i][1]*np.log(good_sample2_num/sample_num)
        # tI1 = -p[:, i][0] *  good_sample1_num / sample_num
        # tI2 = -p[:, i][1] *  good_sample2_num / sample_num
        I1.append(tI1)
        I2.append(tI2)

    # print("I1:", I1)
    # dataframe = pd.DataFrame({'p1': p[0], 'p2': p[1], 'I1': np.array(I1), 'I2':np.array(I2)})
    # dataframe.to_csv("./data_ricean_v2.csv", index=False, sep=',')

    fig = plt.figure() # 创建一个画布figure，然后在这个画布上加各种元素。
    ax = Axes3D(fig) # 将画布作用于 Axes3D 对象上。

    ax.scatter(p[0], p[1],np.array(I1)) # 画出(xs1,ys1,zs1)的散点图。
    # ax.scatter(xs2,ys2,zs2,c='r',marker='^')
    # ax.scatter(xs3,ys3,zs3,c='g',marker='*')

    ax.set_xlabel('X label') # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')

    plt.show()


if __name__ == '__main__':
    # generate_data_sinr()
    # generate_data_rayleign()
    generate_data_ricean()