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
x = x.reshape(len(x),1)
u = np.random.normal(loc=0, scale=0.5, size=100).reshape(100,1)
u = u.reshape(len(u),1)
y_true = (x**2 - 2*x + 10)/10
y = y_true - u
y = y.reshape(len(y))
y_true = y_true.reshape(len(y_true))
x_test = np.sort(np.random.uniform(low=-10, high=10, size=200))
x_test = x_test.reshape(len(x_test), 1)
y_test = (x_test**2 - 2*x_test + 10)/10
y_test = y_test.reshape(len(y_test))

# fit OLS model
ols_model = np.linalg.lstsq(x, y, rcond=-1)[0]
ols_yhat_train = np.sum(x * ols_model, axis=1)
ols_train_errors = np.round(np.sum(np.square(ols_yhat_train - y)) / len(y), decimals=4)
ols_train_risk = np.round(np.sum(np.square(ols_yhat_train - y_true)) / len(y), decimals=4)
ols_yhat_test = np.sum(x_test * ols_model, axis=1)
ols_test_errors = np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)

# fit LSPA model
def lspa_model(n,d, num):
    ncenters = n**(d/(d+4))
    nrestarts = d
    nfinalsteps = n
    return LSPAEstimator(train_args={'ncenters': num, 'nrestarts': nrestarts, 'nfinalsteps': nfinalsteps})

lspa = lspa_model(x.shape[0],x.shape[1], 9)
model_lspa = lspa.train(x, y)
lspa_yhat_train = lspa.predict(model_lspa, x)
lspa_train_risk = np.round(np.sum(np.square(lspa_yhat_train - y_true)) / len(y_true), decimals=4)
lspa_train_error = np.round(np.sum(np.square(lspa_yhat_train - y)) / len(y), decimals=4)
lspa_yhat_test = lspa.predict(model_lspa, x_test)
lspa_test_error = np.round(np.sum(np.square(lspa_yhat_test - y_test)) / len(y_test), decimals=4)

# fit CNLS model
model = CNLS.CNLS(y, x, z=None, cet = CET_ADDI, fun = FUN_COST, rts = RTS_VRS)
model.optimize()
cnls_yhat_train = model.get_frontier()
cnls_train_error = np.round(np.sum(np.square(cnls_yhat_train - y)) / len(y), decimals=4)

plt.figure(11)
plt.scatter(x, y, marker='x',s=50, c='k',label='data points')
plt.plot(x, y_true, markersize=50, c='b',label='true function')
plt.plot(x, lspa_yhat_train, markersize=50, c='r', label='LSPA')
plt.legend()
plt.grid()
plt.xlabel('x', fontsize=20)
plt.ylabel('y=0.1(x^2-2x+10)', fontsize=20)
plt.title('train result(num_centers=6)', fontsize=20)

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
#
# set_random_seed(19)
# x = np.sort(np.random.uniform(low=1, high=10, size=50))
# x = x.reshape(len(x),1)
# u = np.random.normal(loc=0, scale=0.5, size=50).reshape(50,1)
# u = u.reshape(len(u),1)
# y_true = (x**2 - 2*x + 10)/10
# y = y_true - u
# y = y.reshape(len(y))
# y_true = y_true.reshape(len(y_true))
# x_test = np.sort(np.random.uniform(low=1, high=10, size=100))
# x_test = x_test.reshape(len(x_test), 1)
# y_test = (x_test**2 - 2*x_test + 10)/10
# y_test = y_test.reshape(len(y_test))
#
# # fit OLS model
# ols_model = np.linalg.lstsq(x, y, rcond=-1)[0]
# ols_yhat_train = np.sum(x * ols_model, axis=1)
# ols_train_errors = np.round(np.sum(np.square(ols_yhat_train - y)) / len(y), decimals=4)
# ols_train_risk = np.round(np.sum(np.square(ols_yhat_train - y_true)) / len(y), decimals=4)
# ols_yhat_test = np.sum(x_test * ols_model, axis=1)
# ols_test_errors = np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)
#
# # fit LSPA model
# def lspa_model(n,d, num):
#     ncenters = n**(d/(d+4))
#     nrestarts = d
#     nfinalsteps = n
#     return LSPAEstimator(train_args={'ncenters': num, 'nrestarts': nrestarts, 'nfinalsteps': nfinalsteps})
#
# lspa = lspa_model(x.shape[0],x.shape[1], 3)
# model_lspa = lspa.train(x, y)
# lspa_yhat_train = lspa.predict(model_lspa, x)
# lspa_train_risk = np.round(np.sum(np.square(lspa_yhat_train - y_true)) / len(y_true), decimals=4)
# lspa_train_error = np.round(np.sum(np.square(lspa_yhat_train - y)) / len(y), decimals=4)
# lspa_yhat_test = lspa.predict(model_lspa, x_test)
# lspa_test_error = np.round(np.sum(np.square(lspa_yhat_test - y_test)) / len(y_test), decimals=4)
#
# # fit CNLS model
# model = CNLS.CNLS(y, x, z=None, cet = CET_ADDI, fun = FUN_COST, rts = RTS_VRS)
# model.optimize()
# cnls_yhat_train = model.get_frontier()
# cnls_train_error = np.round(np.sum(np.square(cnls_yhat_train - y)) / len(y), decimals=4)
#
# plt.figure(10)
# plt.scatter(x, y, marker='x',s=50, c='k',label='data points')
# plt.plot(x, y_true, markersize=50, c='b',label='true function')
# plt.plot(x, ols_yhat_train, markersize=50, c='g', label='OLS')
# plt.plot(x, lspa_yhat_train, markersize=50, c='r', label='LSPA')
# plt.plot(x, cnls_yhat_train, markersize=50, c='y', label='CNLS')
# plt.legend()
# plt.grid()
#
#
# def paritition_effect_lspa(x, y, y_true):
#     num_centers = np.arange(2, len(x))
#     train_error = []
#     test_error = []
#     for i in num_centers:
#         lspa = LSPAEstimator(train_args={'ncenters': i, 'nrestarts': 5, 'nfinalsteps': x.shape[0]})
#         model = lspa.train(x, y)
#         lspa_yhat_train = lspa.predict(model, x)
#         lspa_train_error = np.round(np.sum(np.square(lspa_yhat_train - y)) / len(y), decimals=4)
#         train_error.append(lspa_train_error)
#         lspa_yhat_test = lspa.predict(model, x_test)
#         lspa_test_error = np.round(np.sum(np.square(lspa_yhat_test - y_test)) / len(y_test), decimals=4)
#         test_error.append(lspa_test_error)
#
#     train_error = np.array(train_error)
#
#     error_change_percent = list(np.diff(train_error) / train_error[0:-1])
#     error_change_percent.insert(0, 0)
#     error_change_percent = np.array(error_change_percent)
#
#     figure_seed = int(np.abs(np.random.randn(1) * 100))
#     plt.figure(figure_seed)
#     plt.plot(num_centers, error_change_percent, label='error change')
#     plt.plot(num_centers, train_error, label='train error')
#     plt.xlabel('number of centers', fontsize=15)
#     plt.ylabel('error change/ train error', fontsize=15)
#     plt.grid()
#     plt.legend()
#     return train_error


plt.show()