import unittest
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#将mydef文件夹加入环境变量
from mydef import *
from mydef import exp, numerical_diff
from mydef import Variable, numerical_diff, exp, log, square, add, sub, mul, div, neg, pow,matmul
from mydef import relu, sigmoid, linear
# 数据集
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x=Variable(x)
y=Variable(y)
# ①权重的初始化
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))
# ②神经网络的推理
def predict(x):
    y = linear(x, W1, b1)
    y = sigmoid(y)
    y = linear(y, W2, b2)
    return y
lr = 0.2
iters = 10000
# ③神经网络的训练
for i in range(iters):
    y_pred = predict(x)
    loss = meansquarederror(y, y_pred)
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()
    W1.data -= lr * W1.grad.data
    print(b1.data)
    print(b1.grad.data)
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0: # 每隔1000次输出一次信息
        print(loss)

# ④画图
import matplotlib.pyplot as plt
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(Variable(t))
plt.plot(t, y_pred.data, color='r')
plt.show()
