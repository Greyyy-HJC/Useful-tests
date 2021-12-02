# %%
#coding=utf-8
import vegas
import math
import gvar as gv
import numpy as np
import scipy.special as ss
from multiprocessing import Process
import timeit
import operator

def main():
    start = timeit.default_timer()
    int_fun = INTFUN(set_para=1)
    integ = vegas.Integrator([[0, 1], [0, 1], [0, 1], [0, 4], [0, 4]])
    result = integ(int_fun.fun, nitn=10, neval=1000000)
    print(result)   
    end = timeit.default_timer()
    print('single processing time:', str(end-start), 's')
    
    start = timeit.default_timer()
    integ = vegas.Integrator([[0, 1], [0, 1], [0, 1], [0, 4], [0, 4]])
    int_fun = INTFUN2(set_kind='Fe')
    result = integ(int_fun.fun_R, nitn=10, neval=1000000)
    print(result)   
    end = timeit.default_timer()
    print('single processing time:', str(end-start), 's')

    start = timeit.default_timer()
    integ = vegas.Integrator([[0, 1], [0, 1], [0, 1], [0, 4], [0, 4]])
    int_fun = INTFUN3(set_kind='Fe')
    result = integ(int_fun.fun_R, nitn=10, neval=1000000)
    print(result)   
    end = timeit.default_timer()
    print('single processing time:', str(end-start), 's')
    

class INTFUN(object):
    def __init__(self, set_para):
        self.y = set_para

    def fun(self, x):#integral function
        return x[1]*x[2]*x[3]*x[4]

# class INTFUN2(object):
#     def __init__(self, set_kind):
#         self.set_kind = set_kind
#         self.kernelFunction_R = {'Fe':self.Fe,}

#     def fun_R(self, x):#integral function
#         return self.kernelFunction_R[self.set_kind](x)
       
#     def Fe(self, x):
#         return x[1]*x[2]*x[3]*x[4]

class INTFUN3(object):
    def __init__(self, set_kind):
        self.set_kind = set_kind
        self.kernelFunction_R = {'Fe':self.Fe,}

    def fun_R(self, x):#integral function
        return self.kernelFunction_R[self.set_kind](x)
       
    def Fe(self,x):
        return self.her(x[0], x[1], x[2], x[3], x[4])
    
    def her(self,x1, x2, x3, b1, bn):
        return x2*x3*b1*bn

class INTFUN2(object):
    def __init__(self, set_kind):
        self.set_kind = set_kind
        if set_kind=='Fe':
            self.fun_R = self.Fe
        # self.kernelFunction_R = {'Fe':self.Fe,}

    # def fun_R(self, x):#integral function
        # return self.kernelFunction_R[self.set_kind](x)
       
    def Fe(self, x):
        return x[1]*x[2]*x[3]*x[4]


if __name__ == "__main__":
    main()
# %%
