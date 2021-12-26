# %%
############### it shows that constant fit with lsqfit will give different mean value from averaging ##################

import gvar as gv
import numpy as np
import lsqfit as lsf

def fcn(x, p):
    return p['a'] + 0 * x

priors = gv.BufferDict()
priors['a'] = gv.gvar(0, 10)

x = np.arange(6)
# mean = [0.8, 1, 1, 1.2, 1.4, 1.3]
mean = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
sdev = [0.2, 0.3, 0.2, 0.5, 0.4, 0.3]
y = [gv.gvar(mean[i], sdev[i]) for i in range(6)]

fit_result = lsf.nonlinear_fit(data=(x, y), prior=priors, fcn=fcn, maxit=10000, svdcut=1e-100, fitter='scipy_least_squares')

print(fit_result.format(100))
print(np.average(mean))
# %%
