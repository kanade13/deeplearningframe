from mydef import *

import numpy as np
import os
import subprocess
'''
def reshape_sum_backward(gy, x_shape, axis, keepdims):
    # 如果 keepdims 为 True，gy 需要与 x 的形状在 sum 操作时的维度保持一致
    if keepdims:
        # 直接返回，因为形状已经匹配
        return gy
    else:
        if axis is None:
            # 如果 gy 是标量，调整为 x 的形状
            return np.broadcast_to(gy, x_shape)  # 将 gy 形状调整为 x 的形状
        # 如果 keepdims 为 False，gy 会少一些维度，需要在 axis 上进行广播
        # 计算需要广播的维度
        gy_shape = list(gy.shape)
        x_shape = list(x_shape)
        
        # 遍历 x_shape 和 gy_shape，扩展 gy 的形状
        for i, ax in enumerate(axis):
            gy_shape.insert(ax, 1)  # 在 axis 维度插入 1，使其可以广播
        
        # 将 gy 调整为与 x_shape 一致的形状
        return gy.reshape(gy_shape).broadcast_to(x_shape)'''
#，sum 函数通常用于按某些轴进行求和操作。在反向传播时，需要将来自下游的梯度（gy）调整成与前向输入（x_shape）兼容的形状。reshape_sum_backward 就是为此目的而设计的。
def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):# 如果需要调整形状：
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy
'''
def sum_to(x, shape):
    """
    求和操作，使数组 x 的形状变为目标形状 shape。例如，若目标形状的某个维度为 1，
    则对该维度进行求和；若目标形状与输入数组的某些维度不同，则对多余的维度求和。
    
    参数:
        x (np.ndarray): 输入数组。
        shape (tuple): 目标形状。
        
    返回:
        np.ndarray: 经过求和后具有目标形状的数组。
    """
    # 检查目标形状是否是输入形状的前缀
    assert len(shape) <= len(x.shape), "目标形状的维度不能超过输入形状的维度"
    for i, s in enumerate(shape):
        assert s == x.shape[i] or s == 1, f"目标形状 {shape} 无法与输入形状 {x.shape} 对齐"
    
    # 计算需要进行求和的维度,收集到一个元组中
    axes_to_sum = tuple(i for i in range(len(x.shape)) if i >= len(shape) or x.shape[i] != shape[i])
    
    # 对指定的维度进行求和
    result = np.sum(x, axis=axes_to_sum, keepdims=True)
    
    # 去掉多余的维度
    return result.reshape(shape)'''
def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def _dot_var(v, verbose=False):#verbose为True时，打印变量的形状和类型
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)
def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt=dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt+=dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt+=dot_edge.format(id(f), id(y()))#y是weakref,所以要加括号
    return txt
def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:#为什么要避免重复添加:因为存在循环引用
            funcs.append(f)
            seen_set.add(f)
    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func=funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.mydef')
    print("tmp_dir:",tmp_dir)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    #调用dot命令
    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    #显示图像
    try:
        from PIL import Image
        img = Image.open(to_file)
        img.show()
    except:
        pass

if __name__ == '__main__':
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y=sum_to(x,(1,3)) 
    y=sum_to(x,(2,1))
    print(x)
    print(y)
    print(y.shape)
    '''def goldstein(x,y):
        return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    x=Variable(np.array(1.0))
    y=Variable(np.array(1.0))
    z=goldstein(x,y)
    z.backward()

    x.name='x'
    y.name='y'
    z.name='z'
    plot_dot_graph(z, verbose=False, to_file='goldstein.png')
    '''