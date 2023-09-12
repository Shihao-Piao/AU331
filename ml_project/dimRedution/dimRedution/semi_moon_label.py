import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from loadData import *
from cutEachStep import *
from DimReduction_moon_v2 import *
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
np.seterr(invalid='ignore')
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
sample_metal , sample_concrete ,  label_metal , label_concrete  = dimReduction(dim = 3)
total_list = []
for i in range(len(sample_metal[0])):
    tmp1 = []
    tmp2 = []
    for j in range(6):
        for number in sample_metal[j][i]:
            tmp1.append(number)
    tmp1.append(0)
    total_list.append(tmp1)

for i in range(len(sample_concrete[0])):
    tmp1 = []
    tmp2 = []
    for j in range(6):
        for number in sample_concrete[j][i]:
            tmp1.append(number)
    tmp1.append(2)
    total_list.append(tmp1)


def load_data(total_list):
    rng=np.random.RandomState(0)
    index=np.arange(len(total_list))
    rng.shuffle(index)
    data, label = np.split(np.array(total_list), indices_or_sections=(18,), axis=1)
    X=data[index]
    Y=label[index]
    n_labeled_points=int(len(Y)/10)
    unlabeled_index=np.arange(len(Y))[n_labeled_points:]

    return X,Y,unlabeled_index

def test_LabelPropagation(*data):
    X,Y,unlabeled_index=data
    Y_train=np.copy(Y)
    Y_train[unlabeled_index]=-1
    cls=LabelPropagation(max_iter=1000,kernel='rbf',gamma=20,tol=0.01)
    #cls = LabelPropagation(max_iter=1000, n_neighbors=20,  kernel='knn')
    cls.fit(X,Y_train.ravel())
    print("Accuracy:%f"%cls.score(X[unlabeled_index],Y[unlabeled_index]))

def test_LabelSpreading(*data):
    X,Y,unlabeled_index=data
    Y_train=np.copy(Y)
    Y_train[unlabeled_index]=-1
    cls=LabelSpreading(max_iter=100,kernel='rbf',gamma=10)
    #cls = LabelPropagation(max_iter=1000, n_neighbors=20, kernel='knn')
    cls.fit(X,Y_train.ravel())
    predicted_labels=cls.transduction_[unlabeled_index]
    true_labels=Y[unlabeled_index]
    print("Accuracy:%f"%metrics.accuracy_score(true_labels,predicted_labels))

X,Y,unlabeled_index=load_data(total_list)
#test_LabelPropagation(X,Y,unlabeled_index)
test_LabelSpreading(X,Y,unlabeled_index)

