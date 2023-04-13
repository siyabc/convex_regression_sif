import numpy as np
from pystoned import CNLS
from pystoned.plot import plot2d
from pystoned.constant import CET_ADDI, FUN_PROD, RTS_VRS, OPT_LOCAL
from matplotlib import pyplot as plt

np.random.seed(0)
x = np.sort(np.random.uniform(low=1, high=10, size=50))
u = np.random.normal(loc=0, scale=0.2, size=50)
y_true = 1 + np.log(x)
y = y_true - u

model = CNLS.CNLS(y, x, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
model.optimize(OPT_LOCAL)
# model.display_residual()
model.display_alpha()
model.display_beta()

plot2d(model, x_select=0, label_name="CNLS", fig_name='CNLS_frontier')
plt.scatter(x,y_true, label='True value')
plt.show()

optimal_value = np.sum(model.get_residual()**2)
print('The optimal objective value is:', optimal_value)
# print('model.I:', model.I)