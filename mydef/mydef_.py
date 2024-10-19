import numpy as np
import warnings

class Variable:
    def __init__(self,  data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError("data must be of type numpy.ndarray")
        self.data= data
        self.grad = None     #y.grad表示对y求导
        self.creator = None
    def set_creator(self, func):
        self.creator = func


    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)#none的时候默认为1(因为是最后一个)
                                                #使grad与data类型一样
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() 
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)
            else:
                break

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) 
        if np.isscalar(y):
            y= np.array(y)
        output = Variable(y)    #numpy中对零维计算可能变为其他类型
        output.set_creator(self)
        self.input=input
        self.output = output
        return output

    def forward(self, x):
        warnings.warn("forwoar() of Function is not implemented yet, use a subclass", stacklevel=2)
        raise NotImplementedError()

    def backward(self, gy):
        warnings.warn("backward() of Function is not implemented yet, use a subclass", stacklevel=2)
        raise NotImplementedError()
    
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):#gy是上游传过来的梯度
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
def exp(x):
    return Exp()(x)
def numerical_diff(f, x, eps=1e-4):  #中心差分近似求数值微分
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
class Square(Function):
    def forward(self, x):
        return x**2
    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy
        return gx
def square(x):
    return Square()(x)

if __name__ == "__main__":
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
