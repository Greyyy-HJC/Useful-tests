# %%
import gvar as gv
import numpy as np


def mellin_moment(x_ls, lc_ls, n):
    x_array = np.array(x_ls)
    zeta_array = 2 * x_array - 1

    ### normalization check ###
    val = []
    for idx in range(len(zeta_array)):
        if zeta_array[idx] >=-1 and zeta_array[idx] <=1:
            val.append(lc_ls[idx].mean)

    ### n order moment ###
    val = []
    for idx in range(len(zeta_array)):
        if zeta_array[idx] >=-1 and zeta_array[idx] <=1:
            val.append( zeta_array[idx]**n * lc_ls[idx] )

    print('>>> mellin moment order '+str(n)+' :')
    print( np.sum(val) * (zeta_array[1]-zeta_array[0]) * 1/2)

    return


jc_x = gv.load('jc_x')
jc_y = gv.load('jc_y')
jun_x = gv.load('jun_x')
jun_y = gv.load('jun_y')

print(jun_x - jc_x) ### same x
for i in range(len(jun_y)): ### same y
    print(jun_y[i])
    print(jc_y[i])
    print('\n')

print(gv.evalcov(jun_y)) ### different cov matrix
print(gv.evalcov(jc_y))

mellin_moment(jc_x, jc_y, 2) ### different moment
mellin_moment(jun_x, jun_y, 2)

# %%
### separate mean and sdev then remake gvar will eliminate all correlation, which is irreversible ###
### but gv.gvar(mean, cov) will preserve correlation ###

print(gv.evalcov(jc_y))

cov = gv.evalcov(jc_y)

me = [val.mean for val in jc_y]
sd = [val.sdev for val in jc_y]

re_gv = gv.gvar(me, sd)
print(gv.evalcov(re_gv))

re_gv = gv.gvar(me, cov)
print(gv.evalcov(re_gv))
# %%
