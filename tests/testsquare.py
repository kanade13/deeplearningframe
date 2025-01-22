import unittest
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#将mydef文件夹加入环境变量
from mydef import *
from mydef import exp, numerical_diff
class SquareTest(unittest.TestCase):
    def test_gradient_check(self):
        # 随机生成多维数组（1D 到 4D）
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))  # 生成随机维度
            x = Variable(np.random.rand(*shape))
            
            # 测试 square 函数
            y = square(x)
            y.backward()
            
            # 测试梯度和数值梯度的接近程度
            num_grad = numerical_diff(square, x)
            flg = np.allclose(x.grad.data, num_grad, atol=1e-6)  # 允许较小的误差
            self.assertTrue(flg)


unittest.main()