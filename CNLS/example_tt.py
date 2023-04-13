#!/usr/bin/env python

from cvxfit import CvxFit
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

#Generate data
N = 100
n = 1

def f_actual(x):
    return sp.sum(x*x)

X = sp.randn(N, n)
Y = sp.array([f_actual(pt) for pt in X])

#Initialize object with 10 affine functions
#with regularization 0.001, and maximum
#number of iterations 20
fit_object = CvxFit(X=X, Y=Y, type='pwl', extra_param=[100, 1, 2000])

#Perform fit
fit_object.fit()

#See training error; repeat fit if high
print 'Training error: ' + str(fit_object.mean_training_error)

#Compare quality of fit at a random point
pt = sp.randn(1, n)
print 'Actual value: ' + str(f_actual(pt))
print 'Approximate value: ' + str(fit_object.evaluate(pt)[0])
res = []
xlin1 = np.linspace(0,10, 100)
t = []
for i in range(100):
    print 'i:', i
    print xlin1[i]
    fit_object.evaluate(np.array([[xlin1[i]]]))
    res.append(fit_object.evaluate(np.array([[xlin1[i]]])) [0])
    t.append(f_actual(xlin1[i]))

plt.plot(xlin1,res)
plt.plot(xlin1, t)
plt.show()