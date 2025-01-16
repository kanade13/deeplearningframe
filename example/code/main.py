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

import config
class kanade(Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #print('input_size',input_size)
        #print('hidden_size',hidden_size)
        self.fc1 = Linear(in_size=input_size, out_size=hidden_size)
        self.fc2 = Linear(in_size=hidden_size, out_size=output_size)

    def forward(self, x):
        h = self.fc1(x)
        #print('h.shape',h.shape)
        h = sigmoid(h)
        y = self.fc2(h)
        print('y.shape',y.shape)
        return y

    def training(self, x, y, epochs=100, lr=0.01):
        optimizer = SGD(lr)
        optimizer.setup(self)
        for epoch in range(epochs):
            y_pred = self.forward(x)

            #如果y_pred和y的shape不一样，需要对y_pred进行reshape
            if y_pred.shape != y.shape:
                y_pred = y_pred.reshape(y.shape)

            #print('y_pred.shape',y_pred.shape)
            #print('y.shape',y.shape)
            loss = meansquarederror(y_pred, y)
            print(loss)
            loss.backward()
            optimizer.update()
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.data}')

        # 评估模型性能
        accuracy,precision,recall,f1_score=evaluate(y_pred, y)
        #f1 = f1_score(y_pred, y)
        print(f'Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1_score}')

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
    lr = 0.2
    max_iters = 10000
    hidden_size = 10
    model = MLP([hidden_size, hidden_size ,2])
    optimizer = SGD(lr).setup(model)
    for i in range(max_iters):
        y_predict = model(training_x)
        loss = meansquarederror(training_y, y_predict)
        loss.backward()
        optimizer.update()
        if i%1000 == 0:
            print(loss)


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
    kanade_model = kanade(config.Config.input_size, config.Config.hidden_size, config.Config.output_size)
    kanade_model.training(Variable(training_x), Variable(training_y), config.Config.num_epochs, config.Config.learning_rate)
    #kanade_model.save_model(config_.Config.save_model_path)
    print('completed!')
