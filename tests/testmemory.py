import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#将mydef文件夹加入环境变量
from mydef import *
import time
from memory_profiler import profile
@profile
def test_memory():
    for i in range(1000):
        x=Variable(np.random.rand(10000)) 
        y=square(square(square(x)))
if __name__ == '__main__':
    test_memory()


#改为弱引用前
'''
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     8     53.5 MiB     53.5 MiB           1   @profile
     9                                         def test_memory():
    10     73.7 MiB  -7373.4 MiB        1001       for i in range(1000):
    11     73.8 MiB  -7245.5 MiB        1000           x=Variable(np.random.rand(10000))
    12     73.7 MiB  -7342.4 MiB        1000           y=square(square(square(x)))
'''

#改为弱引用后
'''
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    8     53.6 MiB     53.6 MiB           1   @profile
    9                                         def test_memory():
10     54.2 MiB      0.0 MiB        1001       for i in range(1000):
11     54.2 MiB      0.1 MiB        1000           x=Variable(np.random.rand(10000))
12     54.2 MiB      0.5 MiB        1000           y=square(square(square(x)))

'''