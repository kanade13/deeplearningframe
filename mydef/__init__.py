from .mydef_ import Variable
from .mydef_ import square
from .mydef_ import exp
from .mydef_ import numerical_diff
from .mydef_ import Square
from .mydef_ import Exp
from .mydef_ import setup_variable
from .config_ import Config
from .mydef_ import Function
from .mydef_ import Transpose
from .mydef_ import Reshape
from .mydef_ import BroadcastTo
from .mydef_ import SumTo
from .mydef_ import Sum
from .mydef_ import MatMul
__all__ = ['Variable', 'square', 'Square', 'Exp','exp','numerical_diff','Function','Config','setup_variable','Transpose','Reshape','BroadcastTo','SumTo','Sum','MatMul']

setup_variable()#设置Variable类的属性
