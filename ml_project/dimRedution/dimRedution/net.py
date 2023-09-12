'''
import numpy as np
t=np.array([
    [[[1,2],[1,3]],1],
    [[1, 2, 3], 2],
])
for i,j in t:
    print(i,j)
print(np.array(t[0][0]).reshape(-1))
'''

from loadData import *
from cutEachStep import *
from jiangwei_together import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn import decomposition
from sklearn import svm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

'''
sample_metal, label = dimReduction(0, '../data/metal_data/')
sample_metal = np.array(sample_metal)
print(sample_metal.shape, label)

sample_water, label = dimReduction(1, '../data/result/')
sample_water = np.array(sample_water)
print(sample_water.shape, label)

sample_concrete, label = dimReduction(2, '../data/concrete_data/')
sample_concrete = np.array(sample_concrete)
print(sample_concrete.shape, label)
'''
sample_metal , sample_concrete , sample_water , label_metal , label_concrete , label_water = dimReduction(dim = 3)

total_list = []
for i in range(len(sample_metal[0])):
    tmp1 = []
    tmp2 = []
    for j in range(6):
        tmp1.append(list(sample_metal[j][i]))
    tmp2.append(tmp1)
    tmp2.append(0)
    total_list.append(tmp2)

for i in range(len(sample_water[0])):
    tmp1 = []
    tmp2 = []
    for j in range(6):
        tmp1.append(list(sample_water[j][i]))
    tmp2.append(tmp1)
    tmp2.append(1)
    total_list.append(tmp2)

for i in range(len(sample_concrete[0])):
    tmp1 = []
    tmp2 = []
    for j in range(6):
        tmp1.append(list(sample_concrete[j][i]))
    tmp2.append(tmp1)
    tmp2.append(2)
    total_list.append(tmp2)

random.shuffle(total_list)

train_list = []
test_list = []

for i in range(int(0.8*len(total_list))):
    train_list.append(total_list[i])
for i in range(int(0.8*len(total_list)),len(total_list)):
    test_list.append(total_list[i])

# 60 20 10
input_size = 18
hidden_size1 = 60
hidden_size2 = 20
hidden_size3 = 10
num_classes = 3
num_epochs = 5
batch_size = 100
learning_rate = 0.001
epoch = 200

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2,hidden_size3, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        #x = torch.from_numpy(x)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = NeuralNet(input_size, hidden_size1,hidden_size2,hidden_size3, num_classes).to(device)
#model = torch.load('\model.pkl')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

s1 = time.time()

for epoch in range(epoch):
        sum_loss = 0.0
        count = 0
        for inputs , label in train_list:
            count+=1
            inputs = np.array(inputs).reshape(-1)

            inputs = torch.tensor(inputs, dtype=torch.float32)
            #inputs = torch.from_numpy(inputs)

            optimizer.zero_grad()#将梯度归零
            outputs = model(inputs)#将数据传入网络进行前向运算
            outputs = torch.unsqueeze(outputs, 0)
            loss = criterion(outputs, torch.tensor([label]))#得到损失函数
            loss.backward()#反向传播
            optimizer.step()#通过梯度做一步参数更新

            # print(loss)
            sum_loss += loss.item()
        print('epoch:%d loss:%.05f' % (epoch + 1, sum_loss / count))
e1 = time.time()
print('train:',e1-s1,'seconds')
model.eval()#将模型变换为测试模式
correct = 0
#total = 0
s2 = time.time()
for inputs,label in test_list:
    inputs = np.array(inputs).reshape(-1)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = model(inputs)
    outputs = torch.unsqueeze(outputs, 0)
    _, predicted = torch.max(outputs, 1)#此处的predicted获取的是最大值的下标
    #total += label.size(0)
    correct += (predicted == label)
#print("correct1: ",correct)
e2 = time.time()
acc = correct.item() / len(test_list)
print("Test acc: %.04f" %(acc))
print("test:",e2-s2,'seconds')
f = open("acc.txt" , "r")
result = f.readline()
f.close()
if acc >= float(result):
    f = open("acc.txt", "w")
    f.write(str(acc))
    f.close()

    torch.save(model, '\model.pkl')
    print('model saved')
