import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lspa_main import LSPAEstimator
from matplotlib import cm
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 16,
    "mathtext.fontset":'stix',
}
rcParams.update(config)


def lspa_model(n,d, num):
    ncenters = n**(d/(d+4))
    nrestarts = d
    nfinalsteps = n
    return LSPAEstimator(train_args={'ncenters': num, 'nrestarts': nrestarts, 'nfinalsteps': nfinalsteps})


def model_training(data_x,data_y, plaine_num=2, coef_name = "test"):
    lspa = lspa_model(data_x.shape[0],data_x.shape[1], plaine_num) #===================================
    model_lspa = lspa.train(data_x, data_y, coef_name)

    xlin1 = np.linspace(min(data_x[:, 0]), max(data_x[:, 0]), 50)
    xlin2 = np.linspace(min(data_x[:, 1]), max(data_x[:, 1]), 50)
    XX0, XX1 = np.meshgrid(xlin1, xlin2)
    
    Z = np.zeros((len(xlin1), len(xlin1)))
    for i in range(len(xlin1)):
        for j in range(len(xlin1)):
            Z[i, j] = -lspa.predict(model_lspa, np.array([[XX0[i, j], XX1[i, j]]]))

    fig = plt.figure()
    ax = Axes3D(fig)
    X = xlin1
    Y = xlin2
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0.5, antialiased=False, alpha=0.7)
    dp = ax.scatter(data_x[:, 0], data_x[:, 1], -data_y, marker='.', s=15, color="#191970", alpha=1)

    ax.set_xlabel("$p_1$", fontsize=20)
    ax.set_ylabel("$p_2$", fontsize=20)
    ax.set_zlabel("$I_2(p)$", fontsize=20)
    ax.xaxis._axinfo["grid"]['color'] = '#DCDCDC'
    ax.yaxis._axinfo["grid"]['color'] = '#DCDCDC'
    ax.zaxis._axinfo["grid"]['color'] = '#DCDCDC'
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.show()
    

def rayleign_fitting_I1(plaine_num=1, coef_name="lspa_coef_rayleign_I1_n"):
    my_data = pd.read_csv('../data_rayleign.csv')
    data_x = np.array([my_data.p1, my_data.p2]).T
    data_y = -np.array([my_data.I1])[0]
    model_training(data_x, data_y, plaine_num=plaine_num,coef_name = coef_name+str(plaine_num))


def rayleign_fitting_I2(plaine_num=1,coef_name = "lspa_coef_rayleign_I2_n"):
    my_data = pd.read_csv('../data_rayleign.csv')
    data_x = np.array([my_data.p1, my_data.p2]).T
    data_y = -np.array([my_data.I2])[0]
    model_training(data_x, data_y, plaine_num=plaine_num,coef_name = coef_name+str(plaine_num))


def ricean_fitting_I1(plaine_num=1,coef_name = "lspa_coef_ricean_I1_n"):
    my_data = pd.read_csv('../data_ricean.csv')
    data_x = np.array([my_data.p1, my_data.p2]).T
    data_y = -np.array([my_data.I1])[0]
    model_training(data_x, data_y, plaine_num=plaine_num,coef_name = coef_name+str(plaine_num))


def ricean_fitting_I2(plaine_num=1,coef_name = "lspa_coef_ricean_I2_n"):
    my_data = pd.read_csv('../data_ricean.csv')
    data_x = np.array([my_data.p1, my_data.p2]).T
    data_y = -np.array([my_data.I2])[0]
    model_training(data_x, data_y, plaine_num=plaine_num, coef_name = coef_name+str(plaine_num))


if __name__ == '__main__':
    # rayleign_fitting_I1(plaine_num=2)
    # rayleign_fitting_I1(plaine_num=4)
    rayleign_fitting_I1(plaine_num=18)
    #
    # rayleign_fitting_I2(plaine_num=2)
    # rayleign_fitting_I2(plaine_num=4)
    rayleign_fitting_I2(plaine_num=18)

    # ricean_fitting_I1(plaine_num=2)
    # ricean_fitting_I1(plaine_num=4)
    # ricean_fitting_I1(plaine_num=8)
    #
    # ricean_fitting_I2(plaine_num=2)
    # ricean_fitting_I2(plaine_num=4)
    # ricean_fitting_I2(plaine_num=8)