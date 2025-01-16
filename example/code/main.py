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
from imblearn.over_sampling import SMOTE  # 用于过采样
from sklearn.metrics import precision_score, recall_score

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

'''class kanade(Model):
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

        x = Variable(PCA(x.data))

        print(x.shape)

        for epoch in range(epochs):
            y_pred = self.forward(x)
            
            
            #如果y_pred和y的shape不一样，需要对y_pred进行reshape
            if y_pred.shape != y.shape:
                y_pred = y_pred.reshape(y.shape)
            #print('y_pred.shape',y_pred.shape)
            #print('y.shape',y.shape)
            loss = meansquarederror(y_pred, y)
            #loss=weighting_mean_square_error()(y_pred,y,w)
            loss.backward()
            optimizer.update()
            
            if epoch%3 == 2:
                lr = lr / 2
                optimizer.lr=lr
            print(f'Epoch {epoch}, Loss: {loss.data}')
        #对y_pred排序
        sorted_pred = np.sort(y_pred.data)[::-1]
        s=sorted_pred[6257]
        print('s:',s)
        for i in range(6250,6300):
            print(sorted_pred[i])
        
        y_pred = self.forward(x)
        print(y_pred)
        y_pred.data = np.where(y_pred.data > -0.5, 1, 0)
        #print(y_pred.data)

        # 评估模型性能
        accuracy,precision,recall,f1_score=evaluate(y_pred, y_0)
        #f1 = f1_score(y_pred, y)
        print(f'Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1_score}')
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(epochs), losses, label="Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.show()
        # 保存模型
        #self.save_model('model_weights.npy')
        

    def save_model(self, filename):
        weights = [param.data for param in self.params()]
        np.save(filename, weights)
'''
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
            w[i] = 1
    kanade_model = kanade(config.Config.input_size, config.Config.hidden_size1,config.Config.hidden_size2,config.Config.output_size)
    kanade_model.training(Variable(training_x), Variable(training_y), config.Config.num_epochs, config.Config.learning_rate,weight=w)
    #kanade_model.save_model(config_.Config.save_model_path)
    print('completed!')'''
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


# 假设你有一个数据集 X (形状: [样本数, 特征数]) 和 y (形状: [样本数])
# 这里用 make_classification 创建一个示例数据集
X, y = training_x, training_y
print(X.shape,y.shape)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 使用SMOTE过采样处理类别不平衡
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#y_test_tensor = torch.tensor(y_test, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 转为浮点型
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)    # 转为浮点型


# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义神经网络结构
# 定义神经网络结构
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from imblearn.over_sampling import SMOTE  # 用于过采样

# 假设你有一个数据集 X (形状: [样本数, 特征数]) 和 y (形状: [样本数])
# 这里用 make_classification 创建一个示例数据集
X, y = make_classification(n_samples=1000, n_features=69, n_informative=30, n_classes=2, 
                            weights=[0.9, 0.1], flip_y=0, random_state=42)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 使用SMOTE过采样处理类别不平衡
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

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
        optimizer.zero_grad()
        outputs = model(inputs)
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


