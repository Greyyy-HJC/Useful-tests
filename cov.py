# %%
import gvar as gv
import numpy as np
from numpy.core.fromnumeric import size
import lsqfit as lsf

# %%
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

def cov_mh(a): #协方差矩阵，a.shape = (conf,time)
    con,time = a.shape
    ave = np.broadcast_to(np.mean(a,0),(con,time)) #将均值广播到全部组态
    cov = a-ave #求偏差
    cov = np.matmul(cov.T,cov) / con #求对应偏差相乘的均值
    return cov #cov.shape = (time,time)

data = np.random.normal(loc=1, size=(5,5))
#print(data)

# %%
data_bs = bootstrap(data, 5000) # more steps, better independence 

mean = np.mean(data_bs, axis=0) 
sdev = np.std(data_bs, axis=0)
middle = np.median(data_bs, axis=0) # bs should use median instead of mean

data_avg = gv.dataset.avg_data(data_bs, bstrap=True)

print(mean)
print(middle)
print(sdev)
print(data_avg) # (median, std)

# %% 
# compare two calc of covariance
cov_m = cov_mh(data_bs)
cov_gv = gv.evalcov(data_avg)
cov_np = np.cov(data_bs, rowvar=False)
print(cov_np - cov_gv) # cov from gv and np are slightly different
print(cov_np - cov_m) # little different, not bad func from minhuan

# %%
data_reconstruct = gv.gvar(middle, cov_gv) # use median and cov can reconstruct the gvar list 
print(data_reconstruct)
print(data_avg)

# %%
# now i wanna know how gvar keep correlation in GVar, or say how could we use gv.evalcov() to extract cov matrix from GVar list

a = np.random.normal(size=25)
b = np.random.normal(loc=1, size=25)
c = np.random.normal(loc=5, size=25)

data = {}
data['a'] = a
data['b'] = b
data['c'] = c

data_avg = gv.dataset.avg_data(data)
cov1 = gv.evalcov(data_avg)

sum = 0
for i in range(25):
    sum += a[i] * b[i]
covab = sum/25 - np.average(a) * np.average(b)

sum = 0
for i in range(25):
    sum += a[i] * c[i]
covac = sum/25 - np.average(a) * np.average(c)

# evalcov: Cov(X, Y) = ( mean(XY) - mean(X)*mean(Y) ) / N, divided by N is because bstrap=False
print(covab/25 - cov1['a', 'b']) 
print(covac/25 - cov1['a', 'c'])

# %%
