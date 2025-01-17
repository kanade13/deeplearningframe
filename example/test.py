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
    def __init__(self,nb=False):
        super().__init__()
        self.layer1 = Linear(out_size=128, in_size=69,nobias=nb)
        self.layer2 = Linear(out_size=64,in_size=128,nobias=nb)
        self.layer3 = Linear(out_size=32,in_size=64,nobias=nb)
        self.output = Linear(out_size=2,in_size=32,nobias=nb)

    def forward(self, x):
        x1 = relu(self.layer1(x))
        if np.isnan(x1.data).any():
            print('x',x)
            print('x1',x1)
            print('-----------------------------------------------------------')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('layer1',self.layer1.showparams())
        x2 = relu(self.layer2(x1))
        if np.isnan(x2.data).any():
            print('x',x1)
            print('x2',x2)
            print('-----------------------------------------------------------')
            print('layer1',self.layer1.showparams())
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')            
            print('layer2',self.layer2.showparams())
        x3 = relu(self.layer3(x2))
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
        

model = SimpleNN(nb=False)

# 定义损失函数和优化器
criterion =Soft_Cross_entropy()
optimizer = SGD(lr=1)
optimizer.setup(model)

# 训练模型
num_epochs = 20
batch_size = 64

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
        '''
        if not np.isnan(loss.data) and not np.isinf(loss.data):
            print(flag,f"Loss: {loss.data:.4f}")
            flag+=1
        else:
            pass'''
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
        print('count',count)
        print('-----------------------------------------------------------')
        model.showgrad()
        model.showparams()
        print('-----------------------------------------------------------')

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


