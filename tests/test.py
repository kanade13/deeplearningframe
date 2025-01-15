import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#将mydef文件夹加入环境变量
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mydef import *
from mydef import Variable
from mydef import MatMul
from mydef import Exp
from mydef import Add,Sum
#from mydef import as_variable

'''
def sphere(x,y):
    return x**2+y**2
x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=sphere(x,y)
z.backward()
print(x.grad,y.grad)


def matyas(x,y):
    return 0.26*(x**2+y**2)-0.48*x*y
x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=matyas(x,y)
z.backward()
print(x.grad,y.grad)


def goldstein(x,y):
    return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=goldstein(x,y)
z.backward()
print(x.grad,y.grad)
'''

'''
def himmelblau(x,y):
    return (x**2+y-11)**2+(x+y**2-7)**2

x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=himmelblau(x,y)
z.backward()
print(x.grad,y.grad)

'''
'''
x=np.array([1,2])
y=np.array([[1,2],[3,4]])
print(x.dot(y))
z=np.array([1,2,3])
print(x.dot(z))'''

'''
x=Variable((np.array(2.0)))
#c=Variable(np.array(3))
y=x**5;
y.backward(create_graph=True)
print(x.grad)

gx=x.grad
x.cleargrad()
gx.backward()
print(x.grad)

x=Variable(np.array([2.0,3.0]))
y=2*x
z=2*x.data
print(y,type(y))
print(z,type(z))
'''

'''
x=Variable(np.array(2.0))
y=x+np.array(3.0)
print(y)

#测试运算
x=Variable(np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]))
y1=Variable(np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]))
y1=y1-x

print(y1)

x=Variable(np.array(2))
y=3.0-x
print(y)
x=Variable(np.array(2.0))
a=square(x)
s1=Square()
s2=Square()
y=add(s1(a),s2(a))
y.backward()
print(y.data)
print(x.grad)

A = Square()
B = Exp()
C = Square()
x= Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(b.data)#b.data=1.2840254166877414
print(y.data)#y.data=1.6487212707001282
print(C.input.data)#C.input.data=b.data=1.2840254166877414
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)  
print(b.grad)#b.grad=2.568050833375483=2*b.data*1.0
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)#3.297442541400256
y.grad = np.array(1.0)
y.backward()
print("grad by automatic backward",x.grad)#3.297442541400256
'''

'''    x = Variable(np.array(3.0))
y = Variable(np.array(2.0))
z = add(square(x), square(y))
z.backward()
print(z.data)#13.0
print(x.grad)#6.0
print(y.grad)#4.0
z.backward()
print(x.grad)#12.0
print(y.grad)#8.0'''

'''
x=Variable(np.array([[1,2,3],[4,5,6]]))
y=Reshape([3,2])(x)
print(y)
#z=x.reshape(6)
#print(z)
#z.backward()
#print(x.grad)
'''
'''
x=Variable(np.array([[2,3],[4,5]]))
y=Variable(np.array([[3],[2]]))
A=MatMul()
B=Square()
#z=B(x)
#print(z)
y=x.T()
print(y)
y.backward()
print(x.grad)

x=Variable(np.array([2,3]))
y=Variable(np.array([4,5]))
A=Add()
z=A(x,y)
    '''
'''
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)
M=MatMul()
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y) # 可以省略
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))
def predict(x):
    y = M(x, W) + b
    return y
def mean_squared_error(x0, x1):
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)
lr = 0.1
iters = 100
for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)
    W.cleargrad()
    b.cleargrad()
    loss.backward()
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)'''

#x=Variable(np.array([1,2]))
#W=Variable(np.array([1,2],[3,4]))
a=[[1,2,3],[4,5,6]]
y=np.sum(a,axis=1,keepdims=False)
print(y)