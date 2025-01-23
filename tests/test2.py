import unittest
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#将mydef文件夹加入环境变量
from mydef import *
from mydef import exp, numerical_diff
from mydef import Variable, numerical_diff, exp, log, square, add, sub, mul, div, neg, pow,matmul
from mydef import relu, sigmoid, linear

class FunctionTest(unittest.TestCase):
    def test_square(self):
        # 随机生成多维数组（1D 到 4D）
        for i in range(50):
            #print('i:',i)
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))  # 生成随机维度
            #print('shape:',shape)
            x = Variable(np.random.rand(*shape))
            
            # 测试 square 函数
            y = square(x)
            y.backward()
            
            # 测试梯度和数值梯度的接近程度
            num_grad = numerical_diff(square, x)
            flg = np.allclose(x.grad.data, num_grad, atol=1e-10)  # 允许较小的误差
            self.assertTrue(flg)
     
    def test_exp(self):
        np.set_printoptions(precision=20)  # 设置打印精度
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(100*np.random.rand(*shape))
            y = exp(x)
            y.backward()
            num_grad = numerical_diff(exp, x,eps=1e-4)
            flg = np.allclose(x.grad.data, num_grad, atol=1e-4)
            self.assertTrue(flg)
        
    def test_log(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape) + 1e-3)  # avoid log(0)
            y = log(x)
            y.backward()
            num_grad = numerical_diff(log, x)
            #print('x.grad',x.grad.data)
            #print("num_grad",num_grad)
            flg = np.allclose(x.grad.data, num_grad, atol=1e-6)
            
            self.assertTrue(flg)

    def test_add(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape))
            y = Variable(np.random.rand(*shape))
            z = add(x, y)
            z.backward()
            num_grad_x = numerical_diff(lambda x: add(x, y), x)
            num_grad_y = numerical_diff(lambda y: add(x, y), y)
            flg_x = np.allclose(x.grad.data, num_grad_x, atol=1e-6)
            flg_y = np.allclose(y.grad.data, num_grad_y, atol=1e-6)
            self.assertTrue(flg_x)
            self.assertTrue(flg_y)

    def test_sub(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape))
            y = Variable(np.random.rand(*shape))
            z = sub(x, y)
            z.backward()
            num_grad_x = numerical_diff(lambda x: sub(x, y), x)
            num_grad_y = numerical_diff(lambda y: sub(x, y), y)
            flg_x = np.allclose(x.grad.data, num_grad_x, atol=1e-6)
            flg_y = np.allclose(y.grad.data, num_grad_y, atol=1e-6)
            self.assertTrue(flg_x)
            self.assertTrue(flg_y)

    def test_mul(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape))
            y = Variable(np.random.rand(*shape))
            z = mul(x, y)
            z.backward()
            num_grad_x = numerical_diff(lambda x: mul(x, y), x)
            num_grad_y = numerical_diff(lambda y: mul(x, y), y)
            flg_x = np.allclose(x.grad.data, num_grad_x, atol=1e-6)
            flg_y = np.allclose(y.grad.data, num_grad_y, atol=1e-6)
            self.assertTrue(flg_x)
            self.assertTrue(flg_y)

    def test_div(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape) + 1e-3)  # avoid division by zero
            y = Variable(np.random.rand(*shape) + 1e-3)
            z = div(x, y)
            z.backward()
            num_grad_x = numerical_diff(lambda x: div(x, y), x)
            num_grad_y = numerical_diff(lambda y: div(x, y), y)
            flg_x = np.allclose(x.grad.data, num_grad_x, atol=1e-6)
            flg_y = np.allclose(y.grad.data, num_grad_y, atol=1e-6)
            self.assertTrue(flg_x)
            self.assertTrue(flg_y)

    def test_neg(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape))
            y = neg(x)
            y.backward()
            num_grad = numerical_diff(neg, x)
            flg = np.allclose(x.grad.data, num_grad, atol=1e-6)
            self.assertTrue(flg)

    def test_pow(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape))
            c = np.random.rand()
            y = pow(x, c)
            y.backward()
            num_grad = numerical_diff(lambda x: pow(x, c), x)
            flg = np.allclose(x.grad.data, num_grad, atol=1e-6)
            self.assertTrue(flg)
    def test_mean_squared_error(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 2)))
            x = Variable(np.random.rand(*shape))
            y = Variable(np.random.rand(*shape))
            #print('x:',x)
            #print('y:',y)
            z = meansquarederror(x, y)
            z.backward()
            #print('x.grad.data:',x.grad.data)
            #print('y.grad.data:',y.grad.data)
            def numerical_diff(f, x, eps=1e-4):
                grad = np.zeros_like(x.data)  # 初始化梯度，形状与 x 一致
                it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    orig_value = x.data[idx]
                    
                    # 计算 f(x+eps)
                    x.data[idx] = orig_value + eps
                    y1 = f(x).data  # 多维数组
                    
                    # 计算 f(x-eps)
                    x.data[idx] = orig_value - eps
                    y0 = f(x).data  # 多维数组
                    
                    # 恢复原值
                    x.data[idx] = orig_value

                    # 求梯度：对多维输出的每个元素取导
                    grad[idx] = np.sum((y1 - y0) / (2 * eps))
                    it.iternext()
                return grad
            num_grad_x = numerical_diff(lambda x: meansquarederror(x, y), x)
            num_grad_y = numerical_diff(lambda y: meansquarederror(x, y), y)
            #print('num_grad_x:',num_grad_x)
            #print('num_grad_y:',num_grad_y)
            flg_x = np.allclose(x.grad.data, num_grad_x, atol=1e-6)
            flg_y = np.allclose(y.grad.data, num_grad_y, atol=1e-6)
            self.assertTrue(flg_x)
            self.assertTrue(flg_y)
    '''
    def test_sum(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape))
            a=np.random.randint(0, len(shape))
            y = Sum(axis=a)(x)
            y.backward()
            num_grad = numerical_diff(Sum(axis=a), x)
            print('x.grad.data:',x.grad.data)
            print('num_grad:',num_grad)
            flg = np.allclose(x.grad.data, num_grad, atol=1e-6)
            self.assertTrue(flg)'''
    def test_matmul(self):
        for i in range(50):
            shape1 = (np.random.randint(1, 10), np.random.randint(1, 10))
            #print('shape1:',shape1)
            shape2 = (shape1[1], np.random.randint(1, 5))
            #print('shape2:',shape2)
            x = Variable(np.random.rand(*shape1))
            y = Variable(np.random.rand(*shape2))
            z = matmul(x, y)
            z.backward()
            def numerical_diff(f, x, eps=1e-4):
                grad = np.zeros_like(x.data)  # 初始化梯度，形状与 x 一致
                it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    orig_value = x.data[idx]
                    
                    # 计算 f(x+eps)
                    x.data[idx] = orig_value + eps
                    y1 = f(x).data  # 多维数组
                    
                    # 计算 f(x-eps)
                    x.data[idx] = orig_value - eps
                    y0 = f(x).data  # 多维数组
                    
                    # 恢复原值
                    x.data[idx] = orig_value

                    # 求梯度：对多维输出的每个元素取导
                    grad[idx] = np.sum((y1 - y0) / (2 * eps))
                    it.iternext()
                return grad
            num_grad_x = numerical_diff(lambda x: matmul(x, y), x)
            num_grad_y = numerical_diff(lambda y: matmul(x, y), y)
            #print('x.grad.data:',x.grad.data)
            #print('num_grad_x:',num_grad_x)
            #print('y.grad.data:',y.grad.data)
            #print('num_grad_y:',num_grad_y)
            flg_x = np.allclose(x.grad.data, num_grad_x, atol=1e-6)
            flg_y = np.allclose(y.grad.data, num_grad_y, atol=1e-6)
            self.assertTrue(flg_x)
            self.assertTrue(flg_y)
    def test_relu(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape))
            y = relu(x)
            y.backward()
            num_grad = numerical_diff(relu, x)
            flg = np.allclose(x.grad.data, num_grad, atol=1e-6)
            self.assertTrue(flg)
    def test_sigmoid(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape))
            y = sigmoid(x)
            y.backward()
            num_grad = numerical_diff(sigmoid, x)
            flg = np.allclose(x.grad.data, num_grad, atol=1e-6)
            self.assertTrue(flg)
    def test_linear(self):
        for i in range(50):
            shapex = (np.random.randint(1, 10), np.random.randint(1, 10))
            #print('shape1:',shape1)
            shapew = (shapex[1], np.random.randint(1, 5))
            #print('shapew:',shapew)
            #print('shapex:',shapex)
            x = Variable(np.random.rand(*shapex))
            W = Variable(np.random.rand(*shapew))
            shapeb=np.dot(x.data,W.data).shape
            b = Variable(np.random.rand(*shapeb))
            y = linear(x,W,b)
            y.backward()
            def numerical_diff(f, x, eps=1e-4):
                grad = np.zeros_like(x.data)  # 初始化梯度，形状与 x 一致
                it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    orig_value = x.data[idx]
                    # 计算 f(x+eps)
                    x.data[idx] = orig_value + eps
                    y1 = f(x).data  # 多维数组
                    
                    # 计算 f(x-eps)
                    x.data[idx] = orig_value - eps
                    y0 = f(x).data  # 多维数组
                    
                    # 恢复原值
                    x.data[idx] = orig_value

                    # 求梯度：对多维输出的每个元素取导
                    grad[idx] = np.sum((y1 - y0) / (2 * eps))
                    it.iternext()
                return grad
            num_grad_x = numerical_diff(lambda x: linear(x,W,b), x)
            num_grad_W = numerical_diff(lambda W: linear(x,W,b), W)
            num_grad_b = numerical_diff(lambda b: linear(x,W,b), b)
            flg_x = np.allclose(x.grad.data, num_grad_x, atol=1e-6)
            flg_W = np.allclose(W.grad.data, num_grad_W, atol=1e-6)
            flg_b = np.allclose(b.grad.data, num_grad_b, atol=1e-6)
            self.assertTrue(flg_x)
            self.assertTrue(flg_W)
            self.assertTrue(flg_b)
    '''
    def test_dropout(self):
        for i in range(50):
            shape = tuple(np.random.randint(1, 5) for _ in range(np.random.randint(1, 5)))
            x = Variable(np.random.rand(*shape))
            y = dropout(x)
            y.backward()
            num_grad = numerical_diff(dropout, x)
            flg = np.allclose(x.grad.data, num_grad, atol=1e-6)
            self.assertTrue(flg)
    '''
    
if __name__ == '__main__':
    unittest.main()