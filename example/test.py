import sys
import os
#将mydef文件夹加入环境变量
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mydef import *
from sklearn.datasets import make_classification
import numpy as np
from mydef import Variable, Linear, Model, SGD, meansquarederror, sigmoid, evaluate, copypositive, PCA
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse

#from mydef import evaluate,meansquarederror,Linear,sigmoid,Variable, SGD, Model
# 将上级目录添加到sys.path，以便可以导入config_.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
np.set_printoptions(threshold=20)
#import config
from sklearn.datasets import make_classification
import numpy as np
import random
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 损失函数（交叉熵损失）
def binary_cross_entropy(y_true, y_pred):
    # 避免数值问题加上一个小常数 epsilon
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 损失函数的导数
def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 损失函数的导数
def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def accuracy(y_true, y_pred):
    correct = np.sum(np.round(y_pred) == y_true)
    total = y_true.shape[0]
    return correct / total

# 计算精确率
def precision(y_true, y_pred):
    tp = np.sum((np.round(y_pred) == 1) & (y_true == 1))
    fp = np.sum((np.round(y_pred) == 1) & (y_true == 0))
    return tp / (tp + fp) if tp + fp > 0 else 0

# 计算召回率
def recall(y_true, y_pred):
    tp = np.sum((np.round(y_pred) == 1) & (y_true == 1))
    fn = np.sum((np.round(y_pred) == 0) & (y_true == 1))
    return tp / (tp + fn) if tp + fn > 0 else 0

def batch_generator(X, y, batch_size):
    for i in range(0, int(len(X)/batch_size)):
        if i+batch_size > X.size:
            continue
        yield X[i:i + batch_size], y[i:i + batch_size]

def calculate_accuracy(y_true, y_pred):
    # 预测值 y_pred 是概率值，选择阈值 0.5 将其转为二分类标签
    y_pred = (y_pred > 0.5).astype(int)
    correct = np.sum(y_pred == y_true)
    total = y_true.shape[0]
    accuracy = correct / total
    return accuracy

def calculate_recall(y_true, y_pred):
    # 预测值 y_pred 是概率值，选择阈值 0.5 将其转为二分类标签
    y_pred = (y_pred > 0.5).astype(int)
    
    # True Positive: 预测为正且真实标签为正
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    # False Negative: 预测为负且真实标签为正
    false_negative = np.sum((y_pred == 0) & (y_true == 1))
    
    # Recall 的计算
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    return recall

def calculate_precision(y_true, y_pred):
    # 预测值 y_pred 是概率值，选择阈值 0.5 将其转为二分类标签
    y_pred = (y_pred > 0.5).astype(int)
    
    # True Positive: 预测为正且真实标签为正
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    # False Negative: 预测为负且真实标签为正
    false_positive = np.sum((y_pred == 1) & (y_true == 0))
    
    # Recall 的计算
    recall = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    return recall

def softmax(z):
    e_z = np.exp(z - np.max(z))  # 为了避免数值溢出，减去最大值
    return e_z / e_z.sum(axis=0, keepdims=True)  # 计算softmax概率

# 交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    # 交叉熵损失函数对每个样本进行计算
    # y_true: 真实标签 [0, 1]，y_pred: softmax 输出
    m = y_true.shape[0]  # 样本数量
    # 防止log(0)，对y_pred进行微小的平滑处理
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 避免log(0)的出现
    # 选择正确类别的概率进行计算
    log_likelihood = -np.log(y_pred[range(len(y_true)), y_true])  # 选择正确类别的log
    return np.mean(log_likelihood)  # 返回平均损失


# Softmax 和 交叉熵损失函数的导数
def softmax_derivative(y_true, y_pred):
    softmax_output = y_pred  # 计算softmax
    # 交叉熵和Softmax组合的梯度公式，y_true是标签向量
    grad = softmax_output.copy()
    grad[range(len(y_true)), y_true] -= 1  # 交叉熵和softmax的组合梯度
    return grad / len(y_true)  # 返回平均梯度


class kanade:
    def __init__(self, input_size, hidden_sizes, output_size,beta = 0):
        self.weights = []
        self.biases = []
        self.beta = beta  # 动量因子
        self.v_w = []  # 存储每层的动量项
        self.v_b = [] 

        # 初始化权重和偏置
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
            self.v_w.append(np.zeros_like(self.weights[i]))  # 初始化动量项为零
            self.v_b.append(np.zeros_like(self.biases[i]))  # 初始化偏置的动量项为零

    def forward(self, X):
        self.z = []  # 线性变换 z = WX + b
        self.a = [X]  # 激活值 a
        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)#1输出
            #a = sigmoid(z) if i < len(self.weights) - 1 else softmax(z)  #2输出 隐藏层用 sigmoid，输出层softmax
            self.z.append(z)
            self.a.append(a)
        return self.a[-1]

    def backward(self, X, y, learning_rate):
        m = X.shape[0]  # 样本数量
        y_pred = self.a[-1]
        y = np.expand_dims(y, axis=1)
        # 计算输出层的梯度
        #delta = softmax_derivative(y, y_pred)#2输出
        delta = binary_cross_entropy_derivative(y, y_pred)

        # 反向传播
        for i in reversed(range(len(self.weights))):
            grad_w = np.dot(self.a[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            # 更新动量项
            self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * grad_w
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * grad_b

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.z[i - 1])

            # 使用动量更新权重和偏置
            self.weights[i] -= learning_rate * self.v_w[i]
            self.biases[i] -= learning_rate * self.v_b[i]


    def train(self, X, y, epochs, learning_rate):
        
        for epoch in range(epochs):
        # Mini-batch训练
            # 训练过程  
            all_preds = []  # 存储所有批次的预测值
            all_labels = []  # 存储所有批次的真实标签           
            running_loss = 0.0
            for X_batch, y_batch in batch_generator(X, y, batch_size):
                # 前向传播
                y_pred = self.forward(X_batch)
                # 计算损失
                #loss = cross_entropy_loss(y_batch, y_pred)#此为2输出
                loss = binary_cross_entropy(y_batch, y_pred)#此为1输出
                # 反向传播
                self.backward(X_batch, y_batch, learning_rate)
                running_loss = running_loss + loss
        # 计算准确率、精确率、召回率
                #all_preds.extend((np.argmax(y_pred, axis = 1)).flatten())#2输出  # 假设输出是概率值
                all_preds.extend(y_pred.flatten())#1输出
                all_labels.extend(y_batch.flatten())  # 真正的标签

            # 计算准确率和召回率
            
            accuracy = calculate_accuracy(np.array(all_labels), np.array(all_preds))
            recall = calculate_recall(np.array(all_labels), np.array(all_preds))
            precision = calculate_precision(np.array(all_labels), np.array(all_preds))
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f},Precision:{precision:.4f}")

    def test(self, X, y):
        all_preds = []
        all_labels = []
        for X_batch, y_batch in batch_generator(X, y, batch_size):
                # 前向传播
            
            y_pred = self.forward(X_batch)
            # 计算损失
            # loss = binary_cross_entropy(y_batch, y_pred)
            # 反向传播
            # self.backward(X_batch, y_batch, learning_rate)
            # y_pred = np.argmax(y_pred, axis = 1)#2输出
        # 计算准确率、精确率、召回率
            all_preds.extend(y_pred.flatten())  # 假设输出是概率值
            all_labels.extend(y_batch.flatten())  # 真正的标签

            # 计算准确率和召回率
        accuracy = calculate_accuracy(np.array(all_labels), np.array(all_preds))
        recall = calculate_recall(np.array(all_labels), np.array(all_preds))
        precision = calculate_precision(np.array(all_labels), np.array(all_preds))
        f1_score = 2 / (1 / recall + 1 / precision)
        print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f},Precision:{precision:.4f},f1-score:{f1_score}")




class dataset():
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def data_collection(self):
        '''
            Parameter
            ---------
            data_path: the input data path, which can be a folder or a npy file 
                  
            return
            ------
            x: the training features, with shape of (N, 69), N is the sample size
            y: the training lables, with shape of (N, 1), N is the sample size
        '''
        def normalization(x):
            def normalize(vector):
                max_vals = np.max(vector, axis=0)
                min_vals = np.min(vector, axis=0)
                normalized_vector = (vector - min_vals) / \
                    (max_vals - min_vals + 1e-3)
                return normalized_vector
            if not np.all(x<=1):
                x = normalize(x)
            return x
        try:
            if self.data_path.endswith('npy'):
                data = np.load(self.data_path, allow_pickle=True).item()
                x = normalization(data['features_list'][0])
                y = data['labels_list'][0]
            else:
                x, y = [], []  # Initialize lists to store features and labels
                for npy_file in os.listdir(self.data_path):
                    npy_file_path = os.path.join(self.data_path, npy_file)
                    data = np.load(npy_file_path, allow_pickle=True).item()
                    x.append(normalization(data['features_list'][0]))
                    y.append(data['labels_list'][0])
                x = np.vstack(x).astype(np.float32) 
                y = np.vstack(y).astype(np.float32).reshape(-1)

            return x, y

        except Exception as e:
            print(f"An error occurred while loading data: {e}")  



class myDataset():
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        # 数据集的大小
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # 根据索引返回特征和标签
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label
class myDataLoader():
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))  # 生成一个索引列表

    def __iter__(self):
        # 每次迭代前根据 shuffle 标志决定是否打乱数据
        if self.shuffle:
            random.shuffle(self.indices)  # 随机打乱索引

        # 返回迭代器
        for i in range(0, len(self.dataset), self.batch_size):
            batch_features = []
            batch_labels = []
            
            # 获取当前批次的索引
            batch_indices = self.indices[i:i + self.batch_size]
            
            for idx in batch_indices:
                feature, label = self.dataset[idx]  # 获取样本
                batch_features.append(feature)
                batch_labels.append(label)
            
            # 手动将特征和标签堆叠成数组（类似于 torch.stack）
            batch_features = np.array(batch_features)  # 转为 ndarray
            batch_labels = np.array(batch_labels)      # 转为 ndarray

            yield batch_features, batch_labels

    def __len__(self):
        # 返回 DataLoader 的长度
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size  # 向上取整
    
parser = argparse.ArgumentParser(description='The training process')
parser.add_argument('--data_path', default='', type=str)

args = parser.parse_args()
#print(os.getcwd())

data_path = "data/data"
data_path = args.data_path if args.data_path else data_path
data = dataset(data_path)
training_x, training_y = data.data_collection()
print(f'the training features of the circuits are: {training_x} with shape of {training_x.shape}')
print(f'the training labels of the circuits are: {training_y} with shape of {training_y.shape}')
# 假设你有一个数据集 X (形状: [样本数, 特征数]) 和 y (形状: [样本数])
# 这里用 make_classification 创建一个示例数据集
X, y = training_x, training_y
y = y.astype(int)
'''
# 假设 X 是数据集，y 是标签
# 计算每个特征的均值和标准差
mean = np.mean(X, axis=0)  # 对每一列计算均值
std = np.std(X, axis=0)    # 对每一列计算标准差

# 手动标准化数据
X_standardized = (X - mean) / std

random_state = 42
np.random.seed(random_state)
'''
# 获取数据集的大小
num_samples = X.shape[0]

# 生成一个随机排列的索引
indices = np.random.permutation(num_samples)

# 划分训练集和测试集的样本数量
test_size = 0.05
test_samples = int(num_samples * test_size)
train_samples = num_samples - test_samples

# 根据随机索引分割数据集
train_indices = indices[:train_samples]
test_indices = indices[train_samples:]

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

X = X_train
y = y_train

# 计算每个类别的样本数量
class_0_count = np.sum(y == 0)
class_1_count = np.sum(y == 1)

# 确定我们要增加的少数类样本数量
# 这里我们使用复制的方式简单地进行过采样
if class_0_count > class_1_count:
    # 少数类是 1，复制少数类样本
    minority_class_samples = X[y == 1]
    minority_class_labels = y[y == 1]
    num_samples_to_generate = class_0_count - class_1_count
else:
    # 少数类是 0，复制少数类样本
    minority_class_samples = X[y == 0]
    minority_class_labels = y[y == 0]
    num_samples_to_generate = class_1_count - class_0_count

# 随机复制少数类样本来增加样本数量
additional_samples = minority_class_samples[np.random.choice(minority_class_samples.shape[0], round(num_samples_to_generate/2), replace=True)]
additional_labels = minority_class_labels[np.random.choice(minority_class_labels.shape[0], round(num_samples_to_generate/2), replace=True)]

# 组合过采样后的数据
X_resampled = np.concatenate([X, additional_samples], axis=0)
y_resampled = np.concatenate([y, additional_labels], axis=0)
X_train = X_resampled
y_train = y_resampled
'''
# 设置随机种子（用于控制结果的可重复性）
random_state = 42
np.random.seed(random_state)

# 获取数据集的大小
num_samples = X_resampled.shape[0]

# 生成一个随机排列的索引
indices = np.random.permutation(num_samples)

# 划分训练集和测试集的样本数量
test_size = 0.2
test_samples = int(num_samples * test_size)
train_samples = num_samples - test_samples

# 根据随机索引分割数据集
train_indices = indices[:train_samples]
test_indices = indices[test_samples:]

X_train = X_resampled[train_indices]
y_train = y_resampled[train_indices]
X_test = X_resampled[test_indices]
y_test = y_resampled[test_indices]
'''

train_dataset = myDataset(X_train, y_train)
#train_dataset = myDataset(training_x, training_y)
test_dataset = myDataset(X_test, y_test)
train_loader = myDataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = myDataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型
'''
class SimpleNN(Model):
    def __init__(self,nb=False):
        super().__init__()
        self.layer1 = Linear(out_size=128, in_size=69,nobias=nb)
        self.layer2 = Linear(out_size=64,in_size=128,nobias=nb)
        self.layer3 = Linear(out_size=32,in_size=64,nobias=nb)
        self.output = Linear(out_size=2,in_size=32,nobias=nb)

    def forward(self, x):
        x = Variable(x)
        x1 = self.layer1(x)
        print('x1:',x1)
        x1 = relu(x1)
        #x1 = sigmoid(self.layer1(x))
        #print('x1:',x1.shape)
        if np.isnan(x1.data).any():
            print('x',x)
            print('x1',x1)
            print('-----------------------------------------------------------')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('layer1',self.layer1.showparams())
        x2 = relu(self.layer2(x1))
        #x2 = sigmoid(self.layer2(x1))
        if np.isnan(x2.data).any():
            print('x',x1)
            print('x2',x2)
            print('-----------------------------------------------------------')
            print('layer1',self.layer1.showparams())
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')            
            print('layer2',self.layer2.showparams())
        x3 = relu(self.layer3(x2))
        #x3 = sigmoid(self.layer3(x2))
        if np.isnan(x3.data).any():
            print('x',x)
            print('layer3',self.layer3.showparams())
        y = self.output(x3)
        if np.isnan(y.data).any():
            print('x3',x3)
            print('y',y)
            print('-----------------------------------------------------------')
            print('output',self.output.showparams())
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('layer3',self.layer3.showparams())
        return y
'''
# 定义损失函数和优化器
#criterion =Soft_Cross_entropy()
#criterion = MeanSquareError()
#criterion = jihuohanshu()
#optimizer = SGD(lr=1)
#optimizer.setup(model)

# 训练模型
num_epochs = 20
batch_size = 200
lam = 1e-6
lr = 2

model = kanade(69,[128,64,32],1)
model.train(X_train, y_train, num_epochs, lr)
model.test(X_test, y_test)
'''
for epoch in range(num_epochs):
    print("epoch",epoch)
    print("--------------------")
    #model.showparams()
    model.show_firstlast_grad()
    print("--------------------")
    model.cleargrads()
    permutation = np.random.permutation(X_train.shape[0])#X_train.shape[0] 表示训练数据集 X_train 的样本数量。通过调用 np.random.permutation(X_train.shape[0])，我们生成了一个长度与训练集样本数量相同的数组，其中包含了所有样本的索引，但这些索引的顺序是随机的。
                                                        #这个随机排列的索引数组通常用于打乱数据集，以确保在训练过程中每个批次的数据都是随机的，从而提高模型的泛化能力。通过这种方式，可以避免模型在训练过程中对数据的顺序产生依赖，从而提高模型的鲁棒性和性能。
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    count=0
    for inputs, labels in train_loader:
        count+=1
        outputs = model(inputs)

        #print('outputs',outputs)
        #print('labels',labels)
        
        loss = criterion(outputs, labels)
        #loss = meansquarederror(y, y_pred)
        for param in model.params():
            loss = loss + regularization(param) * lam
        
        #if not np.isnan(loss.data) and not np.isinf(loss.data):
            #print(flag,f"Loss: {loss.data:.4f}")
            #flag+=1
        #else:
            #pass
            #print(flag,f"Loss: {loss.data:.4f}")
            #print('inputs',inputs)
            #model.showparams()
            #print('outputs',outputs)
            #print('labels',labels)
        loss.backward()

        optimizer.update()

        epoch_loss += loss.data
        #print("epoch",epoch,"loss",loss.data)
        predicted = np.argmax(outputs.data, axis=1)
        total += labels.shape[0]
        correct += np.sum(predicted == labels.data)

        all_preds.extend(predicted)
        all_labels.extend(labels.data)
        # 打印每个参数层的梯度
        #print('count',count)
        #print('-----------------------------------------------------------')
        #model.showgrad()
        #model.showparams()
        #print('-----------------------------------------------------------')

    # 计算 Precision 和 Recall
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    true_positive = np.sum((all_preds == 1) & (all_labels == 1))
    false_positive = np.sum((all_preds == 1) & (all_labels == 0))
    false_negative = np.sum((all_preds == 0) & (all_labels == 1))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train):.4f}, "
          f"Accuracy: {100*correct/total:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}")

# 测试模型
model.cleargrads()
correct = 0
total = 0
all_preds = []
all_labels = []

outputs = model(X_test)
predicted = np.argmax(outputs.data, axis=1)
total += y_test.shape[0]
correct += np.sum(predicted == y_test.data)
all_preds.extend(predicted)
all_labels.extend(y_test.data)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

true_positive = np.sum((all_preds == 1) & (all_labels == 1))
false_positive = np.sum((all_preds == 1) & (all_labels == 0))
false_negative = np.sum((all_preds == 0) & (all_labels == 1))

precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Test Accuracy: {100 * correct / total:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
'''



'''
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
lr = 20
max_iter = 10000
hidden_size = 10
lam = 0.005


x_pred = np.linspace(0,1,100)
x_pred = x_pred[:,np.newaxis]
y_pred = model.forward(x_pred)
print(y_pred.shape)
# 绘制数据点与拟合曲线
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="True Data", color="blue", alpha=0.5)  # 绘制数据点
plt.plot(x_pred, y_pred, label="Fitted Curve", color="red", linewidth=2)  # 绘制拟合曲线
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points and Fitted Curve')
plt.legend()
plt.show()'''
#model = MLP((hidden_size, hidden_size, 1),activation=jihuohanshu)
#optimizer = SGD(lr)
#optimizer.setup(model)
# 或者使用下一行统一进行设置
# optimizer = optimizers.SGD(lr).setup(model)
'''for i in range(max_iter):
    y_pred = model(x)
    loss = meansquarederror(y, y_pred)
    #for param in model.params():
    #    loss = loss + regularization(param) * lam
    model.cleargrads()
    loss.backward()
    optimizer.update()
    if i % 1000 == 0:
        print(loss)
model.showparams()

x_pred = np.linspace(0,1,100)
x_pred = x_pred[:,np.newaxis]
y_pred = model(x_pred)
print(y_pred.shape)
# 绘制数据点与拟合曲线
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="True Data", color="blue", alpha=0.5)  # 绘制数据点
plt.plot(x_pred, y_pred.data, label="Fitted Curve", color="red", linewidth=2)  # 绘制拟合曲线
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points and Fitted Curve')
plt.legend()
plt.show()
'''