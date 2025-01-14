import numpy as np
import math
from mydef import *
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
def sin(x):
    return Sin()(x)

def my_sin(x,threshold=1e-4):#基于泰勒展开的sin求导
    y=0
    for i in range(100000):
        c=(-1)**i*x**(2*i+1)/math.factorial(2*i+1)
        t=c*x**(2*i+1)
        y+=t
        if abs(t)<threshold:
            break
    return y
        