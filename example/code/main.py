import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import sys
#将mydef文件夹加入环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#from mydef import *
#from mydef import evaluate,meansquarederror,Linear,sigmoid,Variable, SGD, Model
# 将上级目录添加到sys.path，以便可以导入config_.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
np.set_printoptions(threshold=20)
#import config

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from mydef import *
import config

class kanade(Model):
    def __init__(self, input_size, hidden_size1,hidden_size2, output_size):
        super().__init__()
        #print('input_size',input_size)
        #print('hidden_size',hidden_size)
        self.fc1 = Linear(in_size=input_size, out_size=hidden_size1)
        self.fc2 = Linear(in_size=hidden_size1, out_size=hidden_size2)
        self.fc3 = Linear(in_size=hidden_size2, out_size=output_size)

    def forward(self, x):
        h = self.fc1(x)
        #print('h.shape',h.shape)
        h = sigmoid(h)
        y = self.fc2(h)
        y = sigmoid(y)
        y = self.fc3(y)
        #print('y.shape',y.shape)
        return y

    def training(self, x, y, epochs=100, lr=0.001,weight=None):
        optimizer = SGD(lr)
        optimizer.setup(self)
        losses = []
        y_pred = None
        x_0 = x
        y_0 = y

        

        print(x.shape)

        for epoch in range(epochs):
            y_pred = self.forward(x)
            
            
            #如果y_pred和y的shape不一样，需要对y_pred进行reshape
            if y_pred.shape != y.shape:
                y_pred = y_pred.reshape(y.shape)
            #print('y_pred.shape',y_pred.shape)
            #print('y.shape',y.shape)
            loss = soft_cross_entropy(y_pred, y)
            #loss=weighting_mean_square_error()(y_pred,y,w)
            loss.backward()
            optimizer.update()
            
            if epoch%3 == 2:
                lr = lr / 2
                optimizer.lr=lr
            print(f'Epoch {epoch}, Loss: {loss.data}')
        #对y_pred排序
        '''sorted_pred = np.sort(y_pred.data)[::-1]
        s=sorted_pred[6257]
        print('s:',s)
        for i in range(6250,6300):
            print(sorted_pred[i])'''
        
        y_pred = self.forward(x)
        print(y_pred)
        y_pred.data = np.where(y_pred.data > -0.5, 1, 0)
        print(y_pred.data)

        # 评估模型性能
        accuracy,precision,recall,f1_score=evaluate(y_pred, y_0)
        #f1 = f1_score(y_pred, y)
        print(f'Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1_score}')
        
        '''
        plt.figure(figsize=(8, 6))
        plt.plot(range(epochs), losses, label="Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.show()'''
        # 保存模型
        #self.save_model('model_weights.npy')
        

    def save_model(self, filename):
        weights = [param.data for param in self.params()]
        np.save(filename, weights)

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
if __name__ == '__main__':
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
    # your implementation

    '''
        After collecting the training dataset, you have to train your classificaiton model
        for example:
            model = Your_model_name(training_x, training_y, other parameters)
            model.training()
        The requirements for the model.training() should include:
            1. the detailed training process
            2. the model performance (recall, precision, f1_score) on the training dataset
            3. save the final model
    '''
    #print(config.Config.input_size)
    #print(config.Config.hidden_size)
    #print(config.Config.output_size)
    '''p=[]
    #设置权重
    positive_num = np.sum(training_y)
    negative_num = len(training_y) - positive_num
    w = np.zeros(len(training_y))
    leny=len(training_y)
    for i in range(leny):
        if training_y[i] == 1:
            p.append(i)
            w[i] = (negative_num / positive_num)/4
        else:
            w[i] = 1'''
    kanade_model = kanade(config.Config.input_size, config.Config.hidden_size1,config.Config.hidden_size2,config.Config.output_size)
    kanade_model.training(Variable(training_x), Variable(training_y), config.Config.num_epochs, config.Config.learning_rate)#,weight=w)
    #kanade_model.save_model(config_.Config.save_model_path)
    print('completed!')
'''
    #复制positive项
    training_y = training_y.reshape(leny,1)
    print(len(p))
    print(p)
    for i in range(len(p)):
        if i==0:
            #计时
            import time
            start = time.time()
        training_x = np.vstack((training_x,training_x[p[i]]))
        training_x = np.vstack((training_x,training_x[p[i]]))
        #print(training_y[i].shape)
        #print(training_y[i].reshape(1,).shape)
        if i==0:
            end = time.time()
            print('time:',end-start)       
        training_y = np.vstack((training_y,training_y[p[i]][0].reshape(1,1)))
        training_y = np.vstack((training_y,training_y[p[i]][0].reshape(1,1)))
        if i==0:
            end = time.time()
            print('time:',end-start)
    yyy=training_y.shape[0]
    training_y=training_y.reshape(yyy,)    
    print("finish preprocessing")
    '''
    #print(training_x.shape)
    #print(training_y.shape)
'''
# 定义神经网络结构
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from mydef import Variable, Linear, Model, SGD, meansquarederror, sigmoid, evaluate, copypositive, PCA

# 假设你有一个数据集 X (形状: [样本数, 特征数]) 和 y (形状: [样本数])
# 这里用 make_classification 创建一个示例数据集
X, y = make_classification(n_samples=1000, n_features=69, n_informative=30, n_classes=2, 
                            weights=[0.9, 0.1], flip_y=0, random_state=42)

# 假设 X 是数据集，y 是标签
# 计算每个特征的均值和标准差
mean = np.mean(X, axis=0)  # 对每一列计算均值
std = np.std(X, axis=0)    # 对每一列计算标准差

# 手动标准化数据
X_standardized = (X - mean) / std

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
additional_samples = minority_class_samples[np.random.choice(minority_class_samples.shape[0], num_samples_to_generate, replace=True)]
additional_labels = minority_class_labels[np.random.choice(minority_class_labels.shape[0], num_samples_to_generate, replace=True)]

# 组合过采样后的数据
X_resampled = np.concatenate([X, additional_samples], axis=0)
y_resampled = np.concatenate([y, additional_labels], axis=0)

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
# 转换为 Variable
X_train = Variable(X_train)
y_train = Variable(y_train)
X_test = Variable(X_test)
y_test = Variable(y_test)'''
train_dataset = myDataset(X_train, y_train)
test_dataset = myDataset(X_test, y_test)
train_loader = myDataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = myDataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型
class SimpleNN(Model):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(out_size=128, in_size=69)
        self.layer2 = Linear(out_size=64,in_size=128)
        self.layer3 = Linear(out_size=32,in_size=64)
        self.output = Linear(out_size=2,in_size=32)

    def forward(self, x):
        x = sigmoid(self.layer1(x))
        x = sigmoid(self.layer2(x))
        x = sigmoid(self.layer3(x))
        x = self.output(x)
        return x

model = SimpleNN()

# 定义损失函数和优化器
criterion = meansquarederror
optimizer = SGD(lr=0.01)

# 训练模型
num_epochs = 20
batch_size = 64

for epoch in range(num_epochs):
    model.cleargrads()
    permutation = np.random.permutation(X_train.shape[0])#X_train.shape[0] 表示训练数据集 X_train 的样本数量。通过调用 np.random.permutation(X_train.shape[0])，我们生成了一个长度与训练集样本数量相同的数组，其中包含了所有样本的索引，但这些索引的顺序是随机的。
                                                        #这个随机排列的索引数组通常用于打乱数据集，以确保在训练过程中每个批次的数据都是随机的，从而提高模型的泛化能力。通过这种方式，可以避免模型在训练过程中对数据的顺序产生依赖，从而提高模型的鲁棒性和性能。
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.update()

        epoch_loss += loss.data
        predicted = np.argmax(outputs.data, axis=1)
        total += labels.shape[0]
        correct += np.sum(predicted == labels.data)

        all_preds.extend(predicted)
        all_labels.extend(labels.data)

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

