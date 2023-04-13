# import packages
from pystoned import CNLSDDF
from pystoned.constant import FUN_PROD, OPT_LOCAL
from pystoned.dataset import load_Finnish_electricity_firm
from pystoned.plot import plot3d
import numpy as np
import matplotlib.pyplot as plt

# import Finnish electricity distribution firms data
data = load_Finnish_electricity_firm(x_select=['OPEX', 'CAPEX'],
                                    y_select=['Energy', 'Length', 'Customers'])

# define and solve the CNLS-DDF model
model = CNLSDDF.CNLSDDF(y=data.y, x=data.x, b=None, fun = FUN_PROD, gx= [1.0, 0.0], gb=None, gy= [0.0, 0.0, 0.0])
model.optimize(OPT_LOCAL)


# display the estimates (alpha, beta, gamma, and residual)
model.display_alpha()
model.display_beta()
model.display_gamma()
model.display_residual()
print("data.y:", data.y)
print("model.y:", model.y)

