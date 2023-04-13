from pystoned import CNLS
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from pystoned.dataset import load_Finnish_electricity_firm
from pystoned.plot import plot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import copy


def my_plot3d(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0.5, antialiased=False, alpha=0.6)
    dp = ax.scatter(X, Y, Z, marker='.', s=15, color="#191970", alpha=1)
    plt.show()

def nonp_model(data_x, data_y, fig_name1):
    model = CNLS.CNLS(y=data_y, x=data_x, z=None,
                        cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)

    model.optimize(OPT_LOCAL)
    model.display_alpha()
    model.display_beta()
    model.display_residual()
    # print("model.y:", model.y)
    print("model.alpha1:", model.get_alpha())
    print("model.display_beta1:", model.get_beta()[1])
    print("get_adjusted_alpha:", model.get_adjusted_alpha())


    plot3d(model, x_select_1=0, x_select_2=1)
    # plt.show()
    optimal_value = np.sum(model.get_residual()**2)
    print('The optimal objective value is:', optimal_value)
    return model.get_beta(), model.get_alpha()


def cnls_iteration(trained_beta1, trained_alpha1, trained_beta2, trained_alpha2):
    p_init = np.array([[101.8,10.8]]) #p_iter: [[0.05586284 0.10974677]]
    p_iter = p_init
    p1_iter_list = []
    p2_iter_list = []
    tol = 10e-7
    err = 1
    iter_num = 0
    while err > tol:
        print("iter_num:",iter_num)
        p_temp = copy.deepcopy(p_iter)
        for i in range(len(trained_alpha1)):
            p1_iter_list.append(np.dot(trained_beta1[i], p_iter.T)+trained_alpha1[i])
            p2_iter_list.append(np.dot(trained_beta2[i], p_iter.T) + trained_alpha2[i])
        p_iter[0][0] = min(p1_iter_list)
        p_iter[0][1] = min(p2_iter_list)
        err = max(np.abs(p_iter[0] - p_temp[0]))
        print("err:", err)
        # print("p_iter:", p_iter)
        # print("p_temp:", p_temp)
        # print("np.abs(p_iter - p_temp):", np.abs(p_iter - p_temp))
        iter_num = iter_num+1

    print("p_iter:", p_iter)






def sinr_fitting():
    my_data = pd.read_csv('../data_sinr.csv')
    data_x = np.array([my_data.p1, my_data.p2]).T
    data_y = np.array([my_data.I1]).T
    nonp_model(data_x, data_y,"cnls_sinr1")
    # data_y = np.array([my_data.I2]).T
    # nonp_model(data_x, data_y, "cnls_sinr2")


def rayleign_fitting():
    my_data = pd.read_csv('../data_rayleign.csv')
    # print("my_data.y:", np.array([my_data.p1, my_data.p2]).T)
    data_x = np.array([my_data.p1, my_data.p2]).T
    data_y = np.array([my_data.I1]).T
    trained_beta1, trained_alpha1 = nonp_model(data_x, data_y,"raylein1")

    data_y = np.array([my_data.I2]).T
    trained_beta2, trained_alpha2 = nonp_model(data_x, data_y, "raylein2")
    cnls_iteration(trained_beta1, trained_alpha1, trained_beta2, trained_alpha2)
    dataframe = pd.DataFrame({'beta11': trained_beta1[:,0], 'beta12': trained_beta1[:,1],'beta21': trained_beta2[:,0], 'beta22': trained_beta2[:,1],
                              'alpha1': trained_alpha1, 'alpha2': trained_alpha2})
    dataframe.to_csv("../cnls_coef_raylein.csv", index=False, sep=',')


def ricean_fitting():
    my_data = pd.read_csv('../data_ricean.csv')
    # print("my_data.y:", np.array([my_data.p1, my_data.p2]).T)
    data_x = np.array([my_data.p1, my_data.p2]).T
    data_y = np.array([my_data.I1]).T
    trained_beta1, trained_alpha1 = nonp_model(data_x, data_y,'ricean1')
    data_y = np.array([my_data.I2]).T
    trained_beta2, trained_alpha2 = nonp_model(data_x, data_y, 'ricean2')
    # cnls_iteration(trained_beta1, trained_alpha1, trained_beta2, trained_alpha2)
    dataframe = pd.DataFrame(
        {'beta11': trained_beta1[:, 0], 'beta12': trained_beta1[:, 1], 'beta21': trained_beta2[:, 0],
         'beta22': trained_beta2[:, 1],
         'alpha1': trained_alpha1, 'alpha2': trained_alpha2})
    dataframe.to_csv("../cnls_coef_ricean.csv", index=False, sep=',')



if __name__ == '__main__':
    # sinr_fitting()
    rayleign_fitting()
    # ricean_fitting()