import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import sys
#将mydef文件夹加入环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mydef import *
from mydef import evaluate,meansquarederror,Linear,sigmoid,Variable, SGD, Model
# 将上级目录添加到sys.path，以便可以导入config_.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
np.set_printoptions(threshold=20)
import config
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
        for epoch in range(epochs):
            y_pred = self.forward(x)

            #如果y_pred和y的shape不一样，需要对y_pred进行reshape
            if y_pred.shape != y.shape:
                y_pred = y_pred.reshape(y.shape)
            #print('y_pred.shape',y_pred.shape)
            #print('y.shape',y.shape)
            #loss = meansquarederror(y_pred, y)
            loss=weighting_mean_square_error()(y_pred,y,w)
            loss.backward()
            optimizer.update()
            '''
            if epoch == 10:
                optimizer.lr = 0.0001
            if epoch == 20:
                optimizer.lr = 0.00005
            if epoch == 30:
                optimizer.lr = 0.00001
            if epoch == 40:
                optimizer.lr = 0.000005'''
            
            '''
            if epoch == 20:
                optimizer.lr = 0.001
            if epoch == 40:
                optimizer.lr = 0.0005
            if epoch == 60:
                optimizer.lr = 0.0001
            if epoch == 80:
                optimizer.lr = 0.00005
            if epoch == 100:
                optimizer.lr = 0.00001'''
            if epoch == 50:
                optimizer.lr = 0.001
            print(f'Epoch {epoch}, Loss: {loss.data}')
        #对y_pred排序
        sorted_pred = np.sort(y_pred.data)[::-1]
        print('sorted_pred.shape:',sorted_pred.shape)
        #统计训练集中的正样本数
        positive_num = int(np.sum(y.data))
        print('positive_num:',positive_num)
        self.threshold = sorted_pred[positive_num]
    
    def evaluate(self, test_x, test_y):
        # 评估模型性能
        y_pred = self.forward(test_x)
        y_pred.data = np.where(y_pred.data >self.threshold, 1, 0)
        accuracy,precision,recall,f1_score=evaluate(y_pred, test_y)
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
    #从中随机挑出10%作为测试集,90%作为训练集
    training_x, test_x, training_y, test_y = train_test_split(training_x, training_y, test_size=0.1, random_state=42)
    # 方法1：单变量选择（互信息）
    '''
    selector = SelectKBest(score_func=mutual_info_classif, k=8)  # 选择20个最佳特征
    selector.fit(training_x, training_y)
    feature_scores = selector.scores_
    top_k_indices = np.argsort(feature_scores)[-8:]  # 得分最高的 k 个特征索引
    # 输出互信息最高的特征维度索引和得分
    print("Top k Feature Indices (1-based):", top_k_indices + 1)
    print("Top k Feature Scores:", feature_scores[top_k_indices])
    '''

# 构造新特征矩阵，仅保留前 k 个特征
    top_k_indices = [10,42,26,58,0,1,2,3]
    X_new = training_x[:, top_k_indices]
    '''
    for i in range(len(feature_scores)):
        print(f'Feature {i}: {feature_scores[i]}')
    '''
    #X_new = selector.fit_transform(training_x, training_y)
    print(X_new.shape)
    #print(config.Config.input_size)
    #print(config.Config.hidden_size)
    #print(config.Config.output_size)
    p=[]
    #设置权重
    positive_num = np.sum(training_y)
    negative_num = len(training_y) - positive_num
    w = np.zeros(len(training_y))
    leny=len(training_y)
    for i in range(leny):
        if training_y[i] == 1:
            p.append(i)
            w[i] = (negative_num / positive_num)/5
        else:
            w[i] = 1
    kanade_model = kanade(config.Config.input_size, config.Config.hidden_size1,config.Config.hidden_size2,config.Config.output_size)
    kanade_model.training(Variable(X_new), Variable(training_y), config.Config.num_epochs, config.Config.learning_rate,weight=w)
    kanade_model.evaluate(Variable(X_new), Variable(training_y))
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


