import numpy as np
import weakref
import warnings
import heapq
import contextlib
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_ import Config, using_config
import utils
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

    #         outputs                                   #存在循环引用,导致变量无法被及时释放,使用弱引用,不会增加实参的
    #           ->                                      #引用计数,当outputs被释放时,变量也会被及时释放
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
    @property
    def T(self):
        return transpose(self)

class Parameter(Variable):
    pass

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def numerical_diff(f, x, eps=1e-10):  #中心差分近似求数值微分
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class Function:
    def __call__(self, *inputs):
        #print("inputs:",inputs)
        # 检查 inputs 是否是列表或元组
        if not isinstance(inputs, (list, tuple)):
            print("inputs:",inputs)
            raise TypeError(f"In {self.__class__.__name__}: inputs must be a list or tuple, got {type(inputs).__name__}")
        
        # 检查 inputs 中的每个元素是否为 Variable
        for i, input in enumerate(inputs):
            if not isinstance(input, Variable):
                print("input:",input)
                raise TypeError(f"In {self.__class__.__name__}: input at index {i} is not a Variable, got {type(input).__name__}")
        inputs=[as_variable(input) for input in inputs]
        x = [x.data for x in inputs]
        #print("xxx:",x)
        #for input in inputs:
        y = self.forward(*x) #解包,相当于self.forward(x[0],x[1],...)
        #print("y1:",y)
        if not isinstance(y, tuple):#如果y不是元组，将其转化为元组
            y = (y,)
        #print("y2:",y)
        outputs=[Variable(as_array(output)) for output in y]
        #print("outputs:",outputs)
        #output = Variable(y)    #numpy中对零维计算可能变为其他类型
        if Config.enable_backprop:
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs=inputs
            self.outputs=[weakref.ref(output) for output in outputs]
        #print("len(outputs):",len(outputs)) 
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
        #x = x - np.max(x)
        return np.exp(x)
    def backward(self, gy):#gy是上游传过来的梯度
        x = self.inputs[0] 
        gx = exp(x) * gy#调用exp,exp调用__call__,__call__中取data后调用forward
        return gx
    
def exp(x):
    return Exp()(x)#这里不用传入self

def log2(base, x):#以base为底的对数
    return np.log(x) / np.log(base)

class Log(Function):
    def forward(self, x):
        x = np.clip(x, 1e-15, None)
        return np.log(x)

    def backward(self, gy):
        x = self.inputs[0]
        x.data = np.clip(x.data ,1e-15, None)
        gx = gy / x
        return gx

def log(x):
    return Log()(x)

class Square(Function):
    def forward(self, x):
        return x**2
    def backward(self, gy):
        #print("gy",gy)
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
        self.x_shape, self.y_shape = x.shape, y.shape
        '''
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("Data x contains NaN or Inf.")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("Data y contains NaN or Inf.")
        if np.any(np.isnan(x*y)) or np.any(np.isinf(x*y)):
            print("Data x*y contains NaN or Inf.")'''
        return x * y
    def backward(self, gy:Variable):
        x, y = self.inputs[0],self.inputs[1]
        if self.x_shape != self.y_shape:
            gx = sum_to(y*gy, self.x_shape)
            gy = sum_to(x*gy, self.y_shape)  
        else:
            gx = y * gy
            gy = x * gy      
        #如果gy不是Variable实例,发出一个警告,并将其转化为Variable实例
        if not isinstance(gy,Variable):
            warnings.warn("gy is not a Variable instance, it will be converted to Variable instance")
            gy=Variable(gy)
        return gx, gy
def mul(x, y):
    x1=as_variable(x)
    y=as_variable(y)
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
        self.x_shape, self.y_shape = x.shape, y.shape
        return x / y
    def backward(self, gy):
        x, y = self.inputs[0],self.inputs[1]
        check([x,y,gy])
        gx = gy / y
        gy = gy * (-x / y ** 2)
        if self.x_shape != self.y_shape:
            gx = sum_to(gx, self.x_shape)
            gy = sum_to(gy, self.y_shape)
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

class Transpose(Function):#transpose仅可用于矩阵转置，不可用于向量，向量实现转置需使用reshape
    def forward(self, x):
        y = np.transpose(x)
        return y
        
    def backward(self, gy):
        gx = transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)

class Sum(Function):
    def __init__(self, axis, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        
    def forward(self, x):
        self.x_shape = x.shape
        y = np.sum(x, axis = self.axis, keepdims = self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)#TODO:ultis.reshape_sum_backward
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
        gx = sum_to(gy, self.x_shape)#书中为ultis.sum_to
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
        y = utils.sum_to(x, self.shape)
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
    m=MatMul()(x, W)
    #print("m:",m)
    return m

class MeanSquareError(Function):
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

class Linearf(Function):
    def forward(self, x, W, b=None):
        #print("x:",x)
        #print("W:",W)
        t =np.dot(x, W)
        #print('t:',t)
        if b is None:
            return t.data
        return (t+b).data
    
    def backward(self, gy):
        W = self.inputs[1]
        x = self.inputs[0]
        b = None if self.inputs[2] is None else self.inputs[2]
        #如果b的形状是(a,),转变为(1,a)
        if b.data is not None and b is not None:
            #print('b:',b)
            if b.ndim == 1:
                b = reshape(b, (1, b.size))
        return matmul(gy,W.T) , matmul(x.T,gy), sum_to(gy, b.shape) if self.inputs[2] is not None and self.inputs[2].data is not None else None

def linear(x, W, b=None):
    #print('linearf:',Linearf()(x, W, b))
    return Linearf()(x, W, b)

class Sigmoid(Function):
    def forward(self, x):
        #print(x)    
        #x = np.clip(x, -50, 50)
        y = 1 / (1+np.exp((-1) * x))
        #print("y:",y)
        return y
    
    def backward(self, gy):
        #print('gy',type(gy.data))
        #print('outputs:',self.outputs,type(self.outputs))
        #outputs = [ref() if isinstance(ref, weakref.ReferenceType) else ref for ref in self.outputs]
        #if None in outputs:
        #    raise ValueError("Some outputs have been garbage collected.")
        
        # 确保 outputs 是数值类型
        #outputs = np.array(outputs, dtype=float)
        x = self.inputs[0].data
        #x = np.clip(x, -50, 50)
        o = 1 / (1+np.exp((-1) * x))
        gx = gy * o * (1-o)
        gx = gx.data
        zero_1 = (gx<1e-3) & (gx>0)
        gx[zero_1] = 1e-3
        zero_2 = (gx>-1e-3) & (gx<0)
        gx[zero_2] = 1e-3
        o = None
        return Variable(gx)
    
def sigmoid(x):
    return Sigmoid()(x)

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

def get_item(x, slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


class SoftMax(Function):#输入为由x_i为每一行排成的矩阵,每个x_i都是一个样本,对应的输出的每一行都是一个y的概率分布
    def forward(self, x, axis = 1):
        x = x - np.max(x)
        y = exp(x)
        sum_y = sum(y, axis = axis, keepdims = True)
        return y / sum_y

    def backward(self, gy):
        raise NotImplementedError()
        #TODO:softmax函数求导
def softmax_simple(x: np.ndarray) -> np.ndarray:
    #x中的值过大时,先进行限制
        # 获取输入的最大值和最小值
    max_x = np.max(x)
    min_x = np.min(x)
    
    # 计算缩放因子
    #scale_factor = 10 / max(abs(max_x)+1e-15, abs(min_x)+1e-15)  # 缩放因子，避免溢出，确保最大值不超过500
    

    # 对输入进行缩放
    #x_scaled = x * scale_factor
    #x_scaled = np.arctan(x) * 20 / np.pi
    x_scaled = x

    # 如果是二维数组（batch情况）
    if x_scaled.ndim == 2:
        x_scaled = x_scaled - np.max(x_scaled, axis=1, keepdims=True)  # 防止溢出，减去每一行的最大值
        y = np.exp(x_scaled)
        sum_y = np.sum(y, axis=1, keepdims=True)
    
    # 如果是一维数组（单个样本情况）
    if x_scaled.ndim == 1:
        x_scaled = x_scaled - np.max(x_scaled)  # 防止溢出，减去最大值
        y = np.exp(x_scaled)
        sum_y = np.sum(y)
    
    # 返回 softmax 结果
    return y / (sum_y + 1e-6)

class Soft_Cross_entropy(Function):#二分类交叉熵
    def forward(self, x, t):#在本框架中t为一个0-1的标签(一个值),所以在backward中我使用了t=(t,1-t)

        '''
        x = np.clip(x, -50, 50)
        x = 1 / (1 + np.exp(-x))
        x = np.clip(x, 1e-15, 1-(1e-15))
        y = t * np.log(x) + (1-t) * np.log(1-x)
        '''
        """
        x: 网络输出的 raw scores（未经过 softmax 处理的值）
        t: 真实标签（目标值）
        """

        a = softmax_simple(x)  # 计算 softmax 概率分布
        a = np.clip(a, 1e-10, 1 - 1e-10)  # 防止 log(0) 的发生
        #print('a:',a)
        #print('a.shape:',a.shape)
        
        
        t=(t,1-t)
        
        #将tuple转化为array
        t=np.array(t)
        #print('t:',t)
        #print('t.shape:',t.shape)
        if (t.dot( np.log(a))).ndim==1:
            y = -np.sum(t * np.log(a))  # 计算交叉熵损失
        else:
            y = -np.sum(t .dot( np.log(a)), axis=1)  # 计算交叉熵损失
        #print('y:',y)
        #print('softmaxentropy:',np.sum(y).shape)
        return np.sum(y)  # 返回总损失

    def backward(self, gy):
        
        x ,t = self.inputs[0], self.inputs[1]
        if isinstance(t, Variable):
            t = t.data
        if isinstance(x, Variable):
            x = x.data
        a=softmax_simple(x)
        #gx =  (-t * 1 / (1+exp(x)) , (1-t) * 1 / (1+exp(-x))) *gy
        t=(t,1-t)
        t=np.array(t).T
        dx = a - t  # 交叉熵损失对输入 x 的梯度：dx = softmax_output - target
        dx *= gy 
        return dx 

def soft_cross_entropy(x, t):
    return Soft_Cross_entropy()(x, t)

def softmax(x):
    return SoftMax()(x)

def softmax_cross_entropy_simple(x, t):
    x, t=as_variable(x), as_variable(t)
    N=x.shape[0]
    
    #p = softmax_simple(x)
    p = sigmoid(x)
    
    #p = np.clip(p.data, 1e-15, 1.0)
    log_p = log(p)
    log_1p = log(1 - p)
    #p=p.reshape(p.size,1)
    y = t * log_p + (1 - t) * log_1p
    
    a=Variable(np.array(0))
    for i in range(y.size):
        a = a - y[i]
    #tlog_p = log_p * []
    #y = (-1) * sum(tlog_p) / N
    return a

class Layer:
    def __init__(self):
        self._params = set()
    
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs,tuple):
            outputs = ((outputs,))
        self.inputs = [weakref.ref(input) for input in inputs]
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs)>1 else outputs[0]

    def forward(self, *inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj
    def showparams(self):
        for param in self.params():
            print(f'{param.name}: {param.data}')
    def showgrad(self):
        for param in self.params():
            print(f'{param.name}: {param.grad}')
    def show_firstlast_grad(self):
        count=0
        for param in self.params():
            count+=1
        for i,param in enumerate(self.params()):
            if i==0:
                print(f'{param.name}: {param.grad}')
            if i==count-1:
                print(f'{param.name}: {param.grad}')
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key : param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

class Model(Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = Linear(hidden_size)
        self.l2 = Linear(out_size)

    def forward(self, x):
        y = sigmoid(self.l1(x))
        y = self.l2(y)
        return y

class MLP(Model):
    def __init__(self, fc_output_sizes, activation=sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, output_size in enumerate(fc_output_sizes):
            layer = Linear(output_size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[ : -1]:
            x = self.activation(l(x))
        return self.layers[-1](x)



class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float64, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name = 'W')
        if self.in_size is not None:
            self.__init__W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.ones((1,self.out_size), dtype=dtype), name = 'b')

    def __init__W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
        #print('w.shape',self.W.data.shape)

    def forward(self, x):

        if self.W.data is None:
            self.in_size = x.shape[1]
            self.__init__W()

        '''if isinstance(x,Variable):
            print("x is Variable")
        if isinstance(x,np.ndarray):
            print("x is ndarray")'''
        y = linear(x, self.W, self.b)
        #y=y.data
        '''
        if isinstance(y,Variable):
            print("y is Variable")
        if isinstance(y,np.ndarray):
            print("y is ndarray")'''
        #y = np.clip(y, -1e20, 1e20)
        #y = dropout(y)
        return y

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        #print('params:',params)
        #预处理(可选)
        #for f in self.hooks:
        #    f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
    def update_one(self, param):
        param.data -= self.lr * param.grad.data

class momentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        param.data -= self.lr * param.grad.data
        # Clip the parameter values to be within the range [-1e20, 1e20]
        param.data = np.clip(param.data, -1e10, 1e10)
    
def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis = 1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))

def evaluate(y, t):#计算precision,recall和f_score
    y, t = as_variable(y), as_variable(t)
    
    pred = y.data
    t = t.data.astype(int)
    #pred = y.data.argmax(axis = 1).reshape(t.shape)
    tp=0
    fp=0
    fn=0
    tn=0
    for i in range(len(y)):
        #print('pred:',pred[i])
        #print('t:',t[i])
        a = np.argmax(pred[i])
        if a == t[i] and a == 1:
            tp = tp + 1
        if a == 0 and t[i] == 1:
            fn = fn + 1
        if a == 1 and t[i] == 0:
            fp = fp + 1
        if a == 0 and t[i] == 0:
            tn = tn + 1
    print("tp=",tp)
    print("fp=",fp)
    print("tn=",tn)
    print("fn=",fn)
    accuracy=(tp+tn)/(tp+fn+fp+tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 1 / (1 / precision + 1 / recall)
    return accuracy,precision, recall, f_score

def characteristic(X, threshold = 0.1):
    # 计算每个特征的方差
    variances = np.var(X.data, axis=0)

    # 设置方差阈值，低于此阈值的特征将被删除
    #threshold = 0.1
    selected_features = variances > threshold

    # 筛选特征
    X_selected = X.data[:, selected_features]

    return Variable(X_selected)

def copypositive(x, y):
    positive_indices = np.where(y.data.astype(int) == 1)[0]
    n_samples_to_add = len(y) - len(positive_indices)  # 需要增加的正样本数量
    new_positive_indices = np.random.choice(positive_indices, n_samples_to_add, replace=True)

    xd = x.data
    yd = y.data
        # 扩展正样本和标签
    X_resampled = np.vstack((xd, xd[new_positive_indices]))
    y_resampled = np.hstack((yd, yd[new_positive_indices]))

    x = Variable(X_resampled)
    y = Variable(y_resampled)

    return x,y

def correlation(X):
        # 计算相关性矩阵
    correlation_matrix = np.corrcoef(X.T)  # 计算特征之间的相关性

    # 设置相关性阈值，去除相关性较高的特征
    threshold = 0.9
    selected_features = np.ones(X.shape[1], dtype=bool)

    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            if abs(correlation_matrix[i, j]) > threshold:
                if selected_features[i]:
                    selected_features[j] = False

    # 筛选特征
    X_selected = X[:, selected_features]

    #print(f"选择的特征索引: {np.where(selected_features)[0]}")
    return X_selected

def calculate_vif(X):
    vif = []
    X_with_intercept = np.c_[np.ones(X.shape[0]), X]  # 增加一列常数项，用于计算截距
    for i in range(X.shape[1]):
        X_other = np.delete(X_with_intercept, i+1, axis=1)  # 删除目标特征，计算其他特征
        y = X_with_intercept[:, i+1]  # 当前特征作为目标
        # 计算该特征的VIF（即拟合回归模型的R²值）
        beta = np.linalg.inv(X_other.T @ X_other) @ X_other.T @ y  # 求解最小二乘解
        residuals = y - X_other @ beta  # 残差
        r2 = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))  # 计算R²
        vif.append(1 / (1 - r2))  # VIF = 1 / (1 - R²)
    return np.array(vif)

def VIF(X):
    # 计算每个特征的VIF
    vif = calculate_vif(X)

    # 设置VIF阈值，去除VIF大于阈值的特征
    threshold = 10
    selected_features = vif < threshold

    # 筛选特征
    X_selected = X[:, selected_features]

    return X_selected

def PCA(X):

    # 假设 X 是原始数据，行表示样本，列表示特征  # 100个样本，5个特征

    # 1. 标准化数据
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_standardized.T)

    # 3. 计算特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(cov_matrix)

    # 4. 按特征值排序
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]

    # 5. 计算累计方差贡献率
    explained_variance = eigvals_sorted / np.sum(eigvals_sorted)
    cumulative_variance = np.cumsum(explained_variance)

    # 6. 选择足够的主成分使累计方差贡献率达到90%
    threshold = 0.90
    k = np.argmax(cumulative_variance >= threshold) + 1

    # 7. 选择前 k 个主成分
    selected_eigvecs = eigvecs_sorted[:, :k]

    # 8. 投影数据到前 k 个主成分
    X_pca = X_standardized.dot(selected_eigvecs)

    return X_pca

class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y
    def backward(self, gy):
        x, = self.inputs
        #print('x:',x)
        mask = x.data > 0
        #print('mask:',mask)
        #print('gy:',gy)
        gx = gy * mask
        #gx = np.nan_to_num(x, nan=0, posinf=1e10, neginf=-1e10)
        return gx
def relu(x):
    return ReLU()(x)

class Config:
    enable_backprop = True
    train = True
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    yield
    setattr(Config, name, old_value)
def test_mode():
    return using_config('train', False)

def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)
    if Config.train:
        xp = x#cuda.get_array_module(x)
        mask = np.random.rand(*x.shape) > dropout_ratio
        
        scale= np.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x

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
    Variable.__getitem__ = get_item

if __name__ == "__main__":
    pass
