# -*- coding: utf-8 -*-
import numpy as np
from loadData import *
from cutEachStep import *
from DimReduction_moon_v2 import *
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
L = []
i = 0
for i in range(len(sample_metal[0])):
    tmp1 = []
    tmp2 = []
    for j in range(6):
        for number in sample_metal[j][i]:
            tmp1.append(number)
    tmp1.append(0)
    if i >=50:
        total_list.append(tmp1)
    else:
        L.append(tmp1)
        i+=1

i = 0
for i in range(len(sample_concrete[0])):
    tmp1 = []
    tmp2 = []
    for j in range(6):
        for number in sample_concrete[j][i]:
            tmp1.append(number)
    tmp1.append(1)
    if i >=50:
        total_list.append(tmp1)
    else:
        L.append(tmp1)
        i+=1

def distEclud(vecA, vecB):
    '''
    输入：向量A和B
    输出：A和B间的欧式距离
    '''
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


def newCent(L):
    '''
    输入：有标签数据集L
    输出：根据L确定初始聚类中心
    '''
    centroids = []
    label_list = np.unique(L[:, -1])
    for i in label_list:
        L_i = L[(L[:, -1]) == i]
        cent_i = np.mean(L_i, 0)
        centroids.append(cent_i[:-1])
    return np.array(centroids)


def semi_kMeans(L, U, distMeas=distEclud, initial_centriod=newCent):
    '''
    输入：有标签数据集L（最后一列为类别标签）、无标签数据集U（无类别标签）
    输出：聚类结果
    '''
    dataSet = np.vstack((L[:, :-1], U))  # 合并L和U
    label_list = np.unique(L[:, -1])
    k = len(label_list)  # L中类别个数
    m = np.shape(dataSet)[0]

    clusterAssment = np.zeros(m)  # 初始化样本的分配
    centroids = initial_centriod(L)  # 确定初始聚类中心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 将每个样本分配给最近的聚类中心
            minDist = np.inf;
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i] != minIndex: clusterChanged = True
            clusterAssment[i] = minIndex
    return clusterAssment

# L的最后一列是类别标签
L = np.array(L)
U , label = np.split(np.array(total_list), indices_or_sections=(18,), axis=1)
clusterResult = semi_kMeans(L, U)
label_lis = []
acc = 0
for i in range(50):
    label_lis.append(0)
for i in range(50):
    label_lis.append(1)
for x in label:
    label_lis.append(x)
for i in range(len(label_lis)):
    if label_lis[i] == clusterResult[i]:
        acc += 1

print("acc:" , acc/len(label_lis))
