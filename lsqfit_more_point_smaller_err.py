# %%

'''
通过调整数据点个数，会发现即便后面增加的数据点误差越来越大，仍然会让拟合结果的误差越来越小。这可以理解为每增加一个数据点，不管这个数据点误差多大，都是对拟合加了一个限制，如果增加的数据点都很好地符合拟合结果，那么多一个限制就会让拟合结果的误差变小。
'''

import gvar as gv
import numpy as np
import lsqfit as lsf
import matplotlib.pyplot as plt

errorb = {"markersize": 5, "mfc": "none", "linestyle": "none", "capsize": 3, "elinewidth": 1} # circle

n = 10 ## 多少个数据点

x_ls = np.arange(1, n)
gv_y_ls = []
for i in range(1, n):
    v = gv.gvar(1, 0.5) * i + 1
    gv_y_ls.append(v)


def fcn(x, p):
    return p['b'] + p['a'] * x

priors = gv.BufferDict()
priors['a'] = gv.gvar(1, 10)
priors['b'] = gv.gvar(1, 10)


fit_result = lsf.nonlinear_fit(data=(x_ls, gv_y_ls), prior=priors, fcn=fcn, maxit=10000, svdcut=1e-100, fitter='scipy_least_squares')

print(fit_result.format(100))

fit_x = np.arange(0, n, 0.1)
fit_y = fcn( fit_x, fit_result.p )

fig, ax=plt.subplots()
ax.errorbar(x_ls, [val.mean for val in gv_y_ls], yerr=[val.sdev for val in gv_y_ls], label='data', fmt='o', **errorb)
ax.fill_between( fit_x, [val.mean + val.sdev for val in fit_y], [val.mean - val.sdev for val in fit_y], color='red', alpha=0.4, label='fit' )
ax.legend()
plt.show()
# %%
