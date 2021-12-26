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
    for yy in range(1,5):
        int_fun = INTFUN(set_para=yy)
        integ = vegas.Integrator([[-1, 1], [0, 1], [0, 1], [0, 1]])
        result = integ(int_fun.fun, nitn=10, neval=10000)
        print(result)   
    end = timeit.default_timer()
    print('single processing time:', str(end-start), 's')

    start = timeit.default_timer()###time start
    p = {}
    for yy in range(1,5):
        def run(yy):
            int_fun = INTFUN(set_para=yy)
            integ = vegas.Integrator([[-1, 1], [0, 1], [0, 1], [0, 1]])
            result = integ(int_fun.fun, nitn=10, neval=10000)
            print(result)
            return result
        p[yy] = Process(target=run,args=(yy,))
    p[1].start()
    p[2].start()
    p[3].start()
    p[4].start() 
    p[1].join()
    p[2].join()
    p[3].join()
    p[4].join()
    end = timeit.default_timer()###time end
    print('single processing time:', str(end-start), 's')


class INTFUN(object):
    def __init__(self, set_para):
        self.y = set_para

    def fun(self, x):#integral function
        temp = (x[0]-0.5)**2 +  (x[1]-0.5)**2 +  (x[2]-0.5)**2 + (x[3]-0.5)**2
        return self.y*math.exp(-temp * 100.) * 1013.2118364296088


if __name__ == "__main__":
    main()