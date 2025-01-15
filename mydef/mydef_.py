import numpy as np
import weakref
import warnings
import heapq

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_ import Config, using_config

class Variable:
    __array_priority__ = 100 #即使运算中包含ndarray实例，也会优先调用Variable实例的方法
    def __init__(self,  data:np.ndarray,name=None):
        if data is not None:
            if not isinstance(data,np.ndarray):
                data=np.array(data)
        self.data= data
        self.grad = None     #y.grad表示对y求导
        self.creator = None
        self.name = name#变量名,用于可视化计算图
        self.generation=0
    
    @property
    def shape(self):
        """variable.shape will return the shape of the variable.data"""
        return self.data.shape
    @property
    def ndim(self):
        """variable.ndim will return the ndim of the variable.data"""
        return self.data.ndim
    @property
    def size(self):
        """variable.size will return the size of the variable.data"""
        return self.data.size
    @property
    def dtype(self):
        """variable.dtype will return the dtype of the variable.data"""
        return self.data.dtype
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n","\n" + " " * 9)
        return "variable(" + p + ")"
    
    
    '''def __mul__(self, other):
        return Mul()(self, other)
    
    
    def __add__(self, other):
        return add(self, other)
    def __radd__(self, other):
        return add(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(self, other)
    def __neg__(self,x):#负号运算符
        return Neg()(x)
    

        
    def __sub__(self, other):
        return Sub()(self, other)
    
    def __rsub__(self, other):#2.0-x  ---->__rsub__(x,2.0)
        other=as_array(other)
        return Sub()(other,self) '''
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    def cleargrad(self):
        self.grad = None

    #backward (upperletter is function, lowerletter is variable)

    #variable和function关系图

    #         outputs                                   #存在循环引用
    #           ->
    #x -> A     ->  a   ->  B    ->  b  ->  C    ->  y
    #  <-       <-
    # input   creator


    def backward(self,retain_grad=False,create_graph=False):
        if self.grad is None:
            #self.grad = np.ones_like(self.data)#none的时候默认为1(因为是最后一个)
                                                #使grad与data类型一样
            self.grad = Variable(np.ones_like(self.data))
        funcs = []
        seen_set = set()
        if self.creator is not None:
            heapq.heappush(funcs, (-self.creator.generation, self.creator))  # generation越大,优先级越高,由于是最小堆,取负号
            seen_set.add(self.creator)
        while funcs:
            #print(funcs)
            _, f = heapq.heappop(funcs)  # 取出优先队列中优先级最高的函数
            gygrad = [output().grad for output in f.outputs]
            #gxgrad = f.backward(*gygrad)
            
            with using_config("enable_backprop", create_graph):
                gxgrad=f.backward(*gygrad)
                if not isinstance(gxgrad, tuple):
                    gxgrad = (gxgrad,)
            
            for x, gx in zip(f.inputs, gxgrad):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # 如果不是 None，说明已经有梯度值，需要将新的梯度 gx 累加到现有的梯度上
                if x.creator is not None and x.creator not in seen_set:
                    #print(x.generation)
                    heapq.heappush(funcs, (-x.creator.generation, x.creator))
                    seen_set.add(x.creator)
            #如果不使用seen_set记录,对于多输入的情况会出现重复计算
            #例如y=add(x,x)的情况,如果不使用seen_set,会出现x.grad=2*x.grad
            #修改了此处使得支持多输入
            
            #TODO:使用优先队列的计算图,初步测试通过,可能需要进一步进行单元测试

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # 释放中间变量(弱引用)的梯度
                    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)
    
    def sum(self, axis=None, keepdims=False):
        return sum(self, axis, keepdims)
    
    def transpose(self):
        return transpose(self)
    
    def T(self):
        return transpose(self)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def numerical_diff(f, x, eps=1e-4):  #中心差分近似求数值微分
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class Function:
    def __call__(self, *inputs):
        inputs=[as_variable(input) for input in inputs]
        x = [x.data for x in inputs]
        #for input in inputs:
        y = self.forward(*x) #解包,相当于self.forward(x[0],x[1],...)
        if not isinstance(y, tuple):#如果y不是元组，将其转化为元组
            y = (y,)
        outputs=[Variable(as_array(output)) for output in y]

        #output = Variable(y)    #numpy中对零维计算可能变为其他类型
        if Config.enable_backprop:
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs=inputs
            self.outputs=[weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]#如果只有一个输出，直接返回元素,否则返回列表
   
    def __lt__(self, other):
        return self.generation < other.generation  #如果 priority 相同且 task 之间未定义默认比较顺序，则两个 (priority, task) 元组之间的比较会报错。
    def forward(self, x:np.ndarray):
        warnings.warn("forwoar() of Function is not implemented yet, use a subclass", stacklevel=2)
        raise NotImplementedError()

    def backward(self, gy):
        warnings.warn("backward() of Function is not implemented yet, use a subclass", stacklevel=2)
        raise NotImplementedError()
        
def check(l: list):
    for idx, i in enumerate(l):
        if not isinstance(i, Variable):
            warnings.warn("input is not a Variable instance, it will be converted to Variable instance")
            l[idx] = Variable(i)  # 修改列表中的实际内容
            return False
    return True

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):#gy是上游传过来的梯度
        x = self.inputs[0] 
        gx = exp(x) * gy#调用exp,exp调用__call__,__call__中取data后调用forward
        return gx
    
def exp(x):
    return Exp()(x)#这里不用传入self


class Square(Function):
    def forward(self, x):
        return x**2
    def backward(self, gy):
        print("gy",gy)
        x = self.inputs[0] #为inputs[0],是正确的
        gx = 2*x*gy
        return gx
def square(x):
    return Square()(x)

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return y
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1
def add(x0, x1):
    return Add()(x0, x1)

class Sub(Function):
    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x - y
    def backward(self, gy):
        gx, gy = gy, gy
        if self.x_shape != self.y_shape:
            gx = sum_to(gx, self.x_shape)
            gy = sum_to(gy, self.y_shape)
        return gx, -gy
def sub(x, y):
    y=as_array(y)
    return Sub()(x, y)
def rsub(x, y):
    y=as_array(y)
    return Sub()(y, x)

class Mul(Function):
    def forward(self, x:np.ndarray, y:np.ndarray):
        return x * y
    def backward(self, gy:Variable):
        x, y = self.inputs[0],self.inputs[1]
        #如果gy不是Variable实例,发出一个警告,并将其转化为Variable实例
        if not isinstance(gy,Variable):
            warnings.warn("gy is not a Variable instance, it will be converted to Variable instance")
            gy=Variable(gy)
        return y * gy, x * gy
def mul(x, y):
    x1=as_variable(x)
    return Mul()(x1, y)#???

class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, gy):
        return -gy
def neg(x): 
    return Neg()(x)

class Div(Function):
    def forward(self, x, y):
        return x / y
    def backward(self, gy):
        x, y = self.inputs[0],self.inputs[1]
        check([x,y,gy])
        gx = gy / y
        gy = gy * (-x / y ** 2)
        return gx,gy
def div(x, y):
    x=as_array(x)
    return Div()(x, y)
def rdiv(x, y):
    y=as_array(y)
    return Div()(y, x)


class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        return x ** self.c
    def backward(self, gy):
        x=self.inputs[0]
        check([x,gy])
        gx = self.c * x ** (self.c - 1) * gy
        return gx
def pow(x, c):
    return Pow(c)(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        y = np.reshape(x, self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
        
    def backward(self, gy):
        gx = transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        
    def foward(self, x):
        self.x_shape = x.shape
        y = np.sum(x, axis = self.axis, keepdims = self.keepdims)
        return y
    
    def backward(self, gy):
        gy = ultis.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)#书中为ultis.reshape_sum_backward
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = ultis.sum_to(gy, self.x_shape)#书中为ultis.sum_to
        return gx
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        y = np.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        y = np.dot(x, W)
        return y
    
    def backward(self, gy):
        x, W=self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx,gW
    
def matmul(x, W):
    return MatMul()(x, W)

def MeanSquareError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self ,gy):
        x0 ,x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff *(2. / len(diff))
        gx1 = -gx0
        return gx0, gx1
    
def meansquarederror(x0 ,x1):
    return MeanSquareError()(x0, x1)

def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y=t+b
    t.data=None
    return y

#linear还没写完
class linear(Function):
    def forward(self, x, W, b=None):
        t = matmul(x, W)
        if b == None:
            return t
        return t+b
    
    #def backward()

class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1+np.exp(-x))
        return y
    
    def backward(self, gy):
        gx = gy * self.outputs * (1-self.outputs)
        return gx
    
def sigmoid(x):
    return Sigmoid()(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    
if __name__ == "__main__":
    pass
