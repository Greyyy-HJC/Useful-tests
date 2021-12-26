# %%
import numpy as np
import gvar as gv
import lsqfit as lsf
import matplotlib.pyplot as plt

def bootstrap(conf_ls, N_re): # used to make random numbers be configs
    N_conf = len(conf_ls)
    conf_re = []
    for times in range(N_re):
        idx_ls = np.random.randint(N_conf, size=N_conf)
        temp = []
        for idx in idx_ls:
            temp.append(conf_ls[idx])
        conf_re.append( np.average(temp, axis=0) )

    return np.array(conf_re)


# %%
t = np.arange(0, 20)

m_jc_cv = gv.load('m_jc_cv')
m_jc_err = gv.load('m_jc_err')

m_jc1_cv = gv.load('m_jc1_cv')
m_jc1_err = gv.load('m_jc1_err')

gv_y = [gv.gvar(m_jc_cv[id], m_jc_err[id]) for id in range(0, 20)]

gv_y1 = [gv.gvar(m_jc1_cv[id], m_jc1_err[id]) for id in range(0, 20)]


# %%
def fcn(x, p):
    return p['m'] + x*0

priors = gv.BufferDict()
priors['m'] = gv.gvar(1, 10)

plt.figure()
plt.errorbar(t, [val.mean for val in gv_y], yerr=[val.sdev for val in gv_y])
plt.ylim([1, 2])
plt.show()

# %%

ini = 6
fin = 12

res = lsf.nonlinear_fit(data=(t[ini:fin], gv_y[ini:fin]), prior=priors, fcn=fcn, maxit=10000, svdcut=1e-100, fitter='scipy_least_squares')

print(res.format(100))



# %%


res1 = lsf.nonlinear_fit(data=(t, gv_y1), prior=priors, fcn=fcn, maxit=10000, svdcut=1e-100, fitter='scipy_least_squares')

print(res1.format(100))


# %%
