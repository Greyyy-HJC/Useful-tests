# %%
import numpy as np
import multiprocessing
from multiprocessing.pool import ThreadPool


def _process(i):
    while True:
        pass


# %%
if __name__ == '__main__':

    max_list = np.arange(4, 40)
    thread_pool = ThreadPool(multiprocessing.cpu_count())
    res = thread_pool.map(_process, max_list)
    print(res)


# %%
