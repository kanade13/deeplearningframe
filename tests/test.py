
# 定义神经网络结构
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
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

from sklearn.metrics import precision_score, recall_score

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"



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



# 假设 X_resampled 是特征数据，y_resampled 是标签数据
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
test_indices = indices[train_samples:]

X_train = X_resampled[train_indices]
y_train = y_resampled[train_indices]
X_test = X_resampled[test_indices]
y_test = y_resampled[test_indices]



# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 神经网络定义
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(69, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.relu(self.layer3(x))
        x = self.output(x)
        #x = torch.sigmoid(self.layer1(x))
        #x = self.dropout(torch.sigmoid(self.layer2(x)))
        #x = torch.sigmoid(self.layer3(x))
        #x = self.output(x)
        return x

model = SimpleNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        #print('inputs.shape',inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        #print('outputs.shape',outputs.shape)
        #inputs.shape torch.Size([64, 69])
        #outputs.shape torch.Size([64, 2])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算 Precision 和 Recall
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    true_positive = ((all_preds == 1) & (all_labels == 1)).sum().item()
    false_positive = ((all_preds == 1) & (all_labels == 0)).sum().item()
    false_negative = ((all_preds == 0) & (all_labels == 1)).sum().item()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
          f"Accuracy: {100*correct/total:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}")

# 测试模型
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = torch.tensor(all_preds)
all_labels = torch.tensor(all_labels)

true_positive = ((all_preds == 1) & (all_labels == 1)).sum().item()
false_positive = ((all_preds == 1) & (all_labels == 0)).sum().item()
false_negative = ((all_preds == 0) & (all_labels == 1)).sum().item()

precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Test Accuracy: {100 * correct / total:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
