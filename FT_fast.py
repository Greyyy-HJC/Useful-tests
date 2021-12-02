# %%
import numpy as np


def sum_ft_slow(x_ls, fx_ls, delta_x, output_k): # coordinate to momentum   ## 19 mins
    ls = []
    for idx in range(len(x_ls)):
        ls.append( delta_x/(2*np.pi) * np.exp(1j * x_ls[idx] * output_k) * fx_ls[idx] )
    val = np.sum(np.array(ls))
    return val


def sum_ft_fast(x_ls, fx_ls, delta_x, output_k): # coordinate to momentum    ## 2.5 mins
    x_ls = np.array(x_ls)
    fx_ls = np.array(fx_ls)
    val = delta_x/(2*np.pi) * np.sum( np.exp(1j * x_ls * output_k) * fx_ls )
    return val

