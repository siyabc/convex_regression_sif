#!/usr/bin/env python
import numpy as np
from cvxfit import CvxFit
import scipy as sp
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def myplot3d():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def my_plot3d(model, x, y, fig_name=None, line_transparent=False, pane_transparent=False):
    """Plot 3d estimated function/frontier

    Args:
        model: The input model for plotting.
        x_select_1 (Integer): The selected x for plotting.
        x_select_2 (Integer): The selected x for plotting.
        fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
        fig_name (String, optional): The name of figure to save. Defaults to None.
        line_transparent (bool, optional): control the transparency of the lines. Defaults to False.
        pane_transparent (bool, optional): control the transparency of the pane. Defaults to False.
    """


    fig = plt.figure()
    ax = Axes3D(fig)
    print 'x[0]:', x[:,0]
    dp = ax.scatter(x[:,0], x[:,1], y, marker='.', s=10)

    # Revise the Z-axis left side
    tmp_planes = ax.zaxis._PLANES
    ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                        tmp_planes[0], tmp_planes[1],
                        tmp_planes[4], tmp_planes[5])

    # make the grid lines transparent
    if line_transparent == False:
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    # make the panes transparent
    if pane_transparent != False:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    xlin1 = np.linspace(min(x[:,0]), max(x[:,1]), 30)
    xlin2 = np.linspace(min(x[:,0]), max(x[:,1]), 30)
    # XX, YY = np.meshgrid(xlin1, xlin2)
    XX0, XX1 = np.meshgrid(xlin1, xlin2)

    ZZ = np.zeros((len(xlin1), len(xlin1)))
    for i in range(len(xlin1)):
        for j in range(len(xlin1)):
            print 'x[:,0][i]:', xlin1[i]
            print 'x[:,1][j]:', xlin2[j]
            ZZ[i, j] = model.evaluate(np.array([[xlin1[i],xlin2[j]]]))

    print("zz:", ZZ)
    print("tzz:", type(ZZ))
    print("szz:", np.shape(ZZ))
    fl = ax.plot_surface(XX0, XX1, ZZ, rstride=1, cstride=1, cmap='viridis',
                         edgecolor='none', alpha=0.5)

    # add x, y, z label
    ax.set_xlabel("Input $x1$")
    ax.set_ylabel("Input $x2$")
    ax.set_zlabel("Output $y$", rotation=0)

    if fig_name == None:
        plt.show()
    else:
        plt.savefig(fig_name)


def lspa_fitting(data_x, data_y, fig_name1, fig_name2,extra_param):
    #Generate data
    N = 10
    n = 2

    def f_actual(x):
        return sp.sum(x * x)

    X = sp.randn(N, n)
    Y = sp.array([f_actual(pt) for pt in X])
    data_y = data_y.T[0]

    print 'X:', X
    print 'data_x:', data_x
    print 'Y:', Y
    print 'data_y:', data_y
    # print 'Y:' + str(Y)

    #Initialize object with 10 affine functions
    #with regularization 0.001, and maximum
    #number of iterations 20
    # fit_object = CvxFit(X=X, Y=Y, type='pwl', extra_param=[10, 0.001, 20])
    fit_object = CvxFit(X=data_x, Y=data_y, type='pwl', extra_param=extra_param)

    #Perform fit
    fit_object.fit()

    #See training error; repeat fit if high
    print 'Training error: ' + str(fit_object.mean_training_error)

    #Compare quality of fit at a random point
    # pt = sp.array([[5]])
    # print 'Actual value: ' + str(f_actual(pt))
    # print 'Approximate value: ' + str(fit_object.evaluate(pt)[0])
    res = fit_object.evaluate(data_x)
#===============================================

    print 'res:', res
    # my_plot3d(fit_object, data_x, data_y, fig_name=None, line_transparent=False, pane_transparent=False)
    fig_name = None
    line_transparent = False
    pane_transparent = False
    fig = plt.figure()
    ax = Axes3D(fig)
    print 'x[0]:', data_x[:, 0]
    dp = ax.scatter(data_x[:, 0], data_x[:, 1], data_y, marker='.', s=10)

    # Revise the Z-axis left side
    tmp_planes = ax.zaxis._PLANES
    ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                        tmp_planes[0], tmp_planes[1],
                        tmp_planes[4], tmp_planes[5])

    # make the grid lines transparent
    if line_transparent == False:
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    # make the panes transparent
    if pane_transparent != False:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    xlin1 = np.linspace(min(data_x[:, 0]), max(data_x[:, 0]), 30)
    xlin2 = np.linspace(min(data_x[:, 1]), max(data_x[:, 1]), 30)
    # XX, YY = np.meshgrid(xlin1, xlin2)
    XX0, XX1 = np.meshgrid(xlin1, xlin2)

    ZZ = np.zeros((len(xlin1), len(xlin1)))
    for i in range(len(xlin1)):
        for j in range(len(xlin1)):
            # print 'x[:,0][i]:', xlin1[i]
            # print 'x[:,1][j]:', xlin2[j]
            ZZ[i, j] = fit_object.evaluate(np.array([[XX0[i, j], XX1[i, j]]]))

    print("zz:", ZZ)
    print("tzz:", type(ZZ))
    print("szz:", np.shape(ZZ))
    fl = ax.plot_surface(XX0, XX1, ZZ, rstride=1, cstride=1, cmap='viridis',
                         edgecolor='none', alpha=0.5)

    # add x, y, z label
    ax.set_xlabel("Input $x1$")
    ax.set_ylabel("Input $x2$")
    ax.set_zlabel("Output $y$", rotation=0)

    if fig_name == None:
        plt.show()
    else:
        plt.savefig(fig_name)

    return fit_object


def sinr_fitting():
    my_data = pd.read_csv('data_sinr.csv')
    # print("my_data.y:", np.array([my_data.p1, my_data.p2]).T)
    data_x = np.array([my_data.p1, my_data.p2]).T
    data_y = np.array([my_data.I1]).T
    # print("data_x:", data_x)
    # print("data_y:", data_y)
    lspa_fitting(data_x, data_y,"sinr1","sinr2",extra_param=[1, 0.001, 200])


def rayleign_fitting():
    my_data = pd.read_csv('data_rayleign.csv')
    # print("my_data.y:", np.array([my_data.p1, my_data.p2]).T)
    data_x = np.array([my_data.p1, my_data.p2]).T
    data_y = 100-np.array([my_data.I1]).T
    # print("data_x:", data_x)
    # print("data_y:", data_y)
    lspa_fitting(data_x, data_y,"raylein1", 'rayleign2',extra_param=[10, 0.1, 100])


def ricean_fitting():
    my_data = pd.read_csv('data_ricean.csv')
    # print("my_data.y:", np.array([my_data.p1, my_data.p2]).T)
    data_x = np.array([my_data.p1, my_data.p2]).T
    data_y = np.array([my_data.I1]).T
    # print("data_x:", data_x)
    # print("data_y:", data_y)
    lspa_fitting(data_x, data_y,'ricean1','ricean2',extra_param=[2, 0.001, 200])


if __name__ == '__main__':
    # sinr_fitting()
    rayleign_fitting()
    # ricean_fitting()