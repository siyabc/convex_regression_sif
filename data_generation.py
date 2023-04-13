import numpy as np
import pandas as pd


def generate_data_sinr():
    data_num = 500
    G = np.array([[0.8,0.12],[0.15,0.84]])
    gamma = np.array([0.2,0.5])
    v = np.array([[0.5],[0.8]])
    p = np.abs(50*np.random.randn(2,data_num))
    # print("np.diag(G):", np.diag(G))
    # print("np.linalg.inv(np.diag(np.diag(G))):", np.linalg.inv(np.diag(np.diag(G))))

    F = np.dot(np.linalg.inv(np.diag(np.diag(G))),(G - np.diag(np.diag(G))))
    print("F:", F)

    A = np.dot(np.diag(gamma),F)
    b = np.dot(np.diag(gamma), v)
    print("A:", A) #A: [[0.     0.03  ][0.0625 0.    ]]
    print("b:", b) #b: [[0.1] [0.4]]
    res = np.dot(A,p)+b
    I1 = res[0]
    I2 = res[1]

    dataframe = pd.DataFrame({'p1': p[0], 'p2': p[1], 'I1': I1, 'I2':I2})
    dataframe.to_csv("./data_sinr.csv", index=False, sep=',')


def generate_data_rayleign():
    data_num = 1000
    # G = np.array([[0.8,0.12],[0.15,1.2]])
    # gamma = np.array([0.2,0.5])
    # v = np.array([[0.5],[0.8]])
    # O_bar =np.array([0.5,0.5])

    G = np.array([[0.8, 0.12], [0.15, 1.2]])
    gamma = np.array([0.12, 0.13])
    v = np.array([[0.25], [0.28]])
    O_bar = np.array([0.2, 0.3])

    p = np.abs(50 * np.random.randn(2, data_num))

    I1 = []
    I2 = []
    for i in range(data_num):
        tI1 = (v[0]+G[0][0] * p[:, i][0]*np.log(1+gamma[0]*G[0][1]*p[:,i][1]/(G[0][0]*p[:, i][0])) )[0]/(G[0][0]*np.log(1/(1-O_bar[0])))
        tI2 = (v[1] + G[1][1] * p[:, i][1] * np.log(1 + gamma[1] * G[1][0] * p[:, i][0] / (G[1][1] * p[:, i][1])))[0] / (
                    G[1][1] * np.log(1 / (1 - O_bar[1])))
        I1.append(tI1)
        I2.append(tI2)
    dataframe = pd.DataFrame({'p1': p[0], 'p2': p[1], 'I1': np.array(I1), 'I2':np.array(I2)})
    dataframe.to_csv("./data_rayleign.csv", index=False, sep=',')


def generate_data_ricean():
    data_num = 500
    G = np.array([[0.8,0.12],[0.15,1.2]])
    gamma = np.array([0.2,0.5])
    v = np.array([[0.5],[0.8]])
    O_bar =np.array([0.5,0.5])
    # G = np.array([[0.8, 0.12], [0.15, 1.2]])
    # gamma = np.array([0.2, 0.3])
    # v = np.array([[0.25], [0.28]])
    # O_bar = np.array([0.2, 0.3])

    p = np.abs(50 * np.random.randn(2, data_num))
    I1 = []
    I2 = []
    sample_num = 2000
    chisquare_df = 20

    for i in range(data_num):
        sample_list1 = []
        sample_list2 = []

        for j in range(sample_num):
            h11 = np.random.chisquare(df=chisquare_df, size=1)
            h12 = np.random.chisquare(df=chisquare_df, size=1)
            h21 = np.random.chisquare(df=chisquare_df, size=1)
            h22 = np.random.chisquare(df=chisquare_df, size=1)
            t1 = gamma[0]*(G[0][1]*h12*p[:,i][1]+v[0])/(G[0][0]*h11)
            sample_list1.append(t1[0])
            # print("t1:", t1[0])
            # print("sample_list1:", sample_list1)
            t2 = gamma[1] * (G[1][0] * h21*p[:,i][0] + v[1]) / (G[1][1] * h22)
            sample_list2.append(t2[0])
        tI1 = np.percentile(np.array(sample_list1), (1-O_bar[0])*100)
        tI2 = np.percentile(np.array(sample_list2), (1-O_bar[1]) * 100)
        I1.append(tI1)
        I2.append(tI2)

    # print("I1:", I1)
    dataframe = pd.DataFrame({'p1': p[0], 'p2': p[1], 'I1': np.array(I1), 'I2':np.array(I2)})
    dataframe.to_csv("./data_ricean.csv", index=False, sep=',')

if __name__ == '__main__':
    # generate_data_sinr()
    generate_data_rayleign()
    # generate_data_ricean()