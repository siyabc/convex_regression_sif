import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import OrderedDict

from regression import max_affine_predict, max_affine_fit_partition
from lspa_main import LSPAEstimator

from pystoned import CNLS
from pystoned.plot import plot2d
from pystoned.constant import CET_ADDI, FUN_COST, RTS_VRS, OPT_LOCAL, FUN_PROD, RTS_CRS
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator


# random initialization
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(random.randint(0, 1e8))

seed_limit = 1e6
global_random_seed = 100 + int(np.round((time.time() % 1) * seed_limit))
set_random_seed(global_random_seed)
setup_random_seed = np.random.randint(seed_limit)
data_random_seed = np.random.randint(seed_limit)
training_random_seed = np.random.randint(seed_limit)
testing_random_seed = np.random.randint(seed_limit)

# parameters initialization
nruns = 10  # number of experiment runs
ntestsamples = int(1e6)  # number of test samples to generate

parallel_nworkers = 1
parallel_backend = 'multiprocessing'

set_random_seed(19)
x = np.sort(np.random.uniform(low=-10, high=10, size=100))
x = np.vstack((x,x)).T

# x = x.reshape(len(x),1)
u = np.random.normal(loc=0, scale=0.5, size=100).reshape(100,1)
u = u.reshape(len(u),1)
y_true = (x**2 - 2*x + 10)/10
# y = y_true - u
# y = y.reshape(len(y))
y = y_true[:,0]
print("sd")

my_data = pd.read_csv('data_rayleign.csv')
    # print("my_data.y:", np.array([my_data.p1, my_data.p2]).T)
data_x = np.array([my_data.p1, my_data.p2]).T
data_y = -np.array([my_data.I1])[0]
# print("data_x:", data_x)
# print("data_y:", data_y)
x = data_x
y = data_y




# fit LSPA model
def lspa_model(n,d, num):
    ncenters = n**(d/(d+4))
    nrestarts = d
    nfinalsteps = n
    return LSPAEstimator(train_args={'ncenters': num, 'nrestarts': nrestarts, 'nfinalsteps': nfinalsteps})
print("x.shape[0]:", x.shape[0])
print("x.shape[1]:", x.shape[1])
print("x:", x)
print("y:", y)
lspa = lspa_model(x.shape[0],x.shape[1], 8) #===================================
model_lspa = lspa.train(x, y)
lspa_yhat_train = lspa.predict(model_lspa, x)
# lspa_train_risk = np.round(np.sum(np.square(lspa_yhat_train - y_true)) / len(y_true), decimals=4)
# lspa_train_error = np.round(np.sum(np.square(lspa_yhat_train - y)) / len(y), decimals=4)
# lspa_yhat_test = lspa.predict(model_lspa, x_test)
# lspa_test_error = np.round(np.sum(np.square(lspa_yhat_test - y_test)) / len(y_test), decimals=4)

# fit CNLS model
model = CNLS.CNLS(y, x, z=None, cet = CET_ADDI, fun = FUN_COST, rts = RTS_VRS)
model.optimize()
cnls_yhat_train = model.get_frontier()
cnls_train_error = np.round(np.sum(np.square(cnls_yhat_train - y)) / len(y), decimals=4)


# plt.figure(11)
# plt.scatter(x[:,0], y, marker='x',s=50, c='k',label='data points')
# # plt.plot(x[:,0], y_true, markersize=50, c='b',label='true function')
# plt.plot(x[:,0], lspa_yhat_train, markersize=50, c='r', label='LSPA')
# plt.legend()
# plt.grid()
# plt.xlabel('x', fontsize=20)
# plt.ylabel('y=0.1(x^2-2x+10)', fontsize=20)
# plt.title('train result(num_centers=6)', fontsize=20)








fig_name = None
line_transparent = False
pane_transparent = False
fig = plt.figure()
ax = Axes3D(fig)
dp = ax.scatter(data_x[:, 0], data_x[:, 1], -data_y, marker='.', s=10)

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
        # ZZ[i, j] = fit_object.evaluate(np.array([[XX0[i, j], XX1[i, j]]]))
        ZZ[i, j] = -lspa.predict(model_lspa, np.array([[XX0[i, j], XX1[i, j]]]))

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


















# plt.figure(12)
# plt.scatter(x_test, y_test, marker='x',s=50, c='k',label='data points')
# plt.plot(x_test, y_test, markersize=50, c='b',label='true function')
# plt.plot(x_test, lspa_yhat_test, markersize=50, c='r', label='LSPA')
# plt.plot(x_test, ols_yhat_test, markersize=50, c='g', label='OLS')
# plt.legend()
# plt.grid()
# plt.xlabel('x', fontsize=20)
# plt.ylabel('y=0.1(x^2-2x+10)',fontsize=15)
# plt.title('test result', fontsize=15)

plt.show()