import contextlib

class Config:
    enable_backprop = True#true表示启用反向传播模式


@contextlib.contextmanager
def using_config(name,value):
    old_value = getattr(Config,name)#动态地访问和修改对象的属性
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)

def no_grad():
    return using_config('enable_backprop',False)