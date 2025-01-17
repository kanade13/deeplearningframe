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
from .mydef_ import MatMul,Add,MeanSquareError,Linear,Layer,Log,meansquarederror,Sigmoid
from .mydef_ import GetItem,GetItemGrad,Optimizer,SGD,relu,Soft_Cross_entropy
from .mydef_ import MLP,TwoLayerNet,VIF,PCA,Config,dropout,soft_cross_entropy
from .mydef_ import Mul,Div,Pow,Linearf,linear_simple,softmax_cross_entropy_simple
from .mydef_ import evaluate,Model,sigmoid,characteristic,copypositive,correlation
from .mydef_ import Soft_Cross_entropy,softmax_simple
#from .mydef_ import weighting_mean_square_error
__all__ = ['softmax_simple','Soft_Cross_entropy','soft_cross_entropy','relu','dropout','Config','PCA','VIF','correlation','copypositive','characteristic','sigmoid','Model','evaluate','softmax_cross_entropy_simple','linear_simple','Linearf','Div','Mul','Log','Linear','MeanSquareError','Add','Variable', 'square', 'Square', 'Exp','exp','numerical_diff','Function','Config','setup_variable','Transpose','Reshape','BroadcastTo','SumTo','Sum','MatMul','meansquarederror','Sigmoid','GetItem','GetItemGrad','Optimizer','SGD','MLP','TwoLayerNet','Pow']
'weighting_mean_square_error'
setup_variable()#设置Variable类的属性
