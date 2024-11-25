import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#将mydef文件夹加入环境变量
from mydef import *

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


def himmelblau(x,y):
    return (x**2+y-11)**2+(x+y**2-7)**2
x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=himmelblau(x,y)
z.backward()
print(x.grad,y.grad)




















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

