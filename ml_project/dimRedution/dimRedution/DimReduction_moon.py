from loadData import *
from cutEachStep import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn import decomposition
from sklearn import svm
onehot = \
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

notonehot = ['water', 'metal', 'concrete']


def dimReduction(dim=3.):  ## dim means the dimention to be reduced
    path = '../data/moon-terrain/'  # terrain 路径
    sample_terrain, label_terrain = cut_data(data=3, path=path)
    path = '../data/moon_metal_data/'  # metal 路径
    sample_metal, label_metal = cut_data(data=4, path=path)
    # path = '../data/concrete_data/'  # concrete 路径
    # sample_concrete, label_concrete = cut_data(data=2, path=path)
    ## 转化为dataframe
    # df = pd.DataFrame(sample)
    # df = df.T
    # df.rename(columns={0: 'force_x', 1: 'force_y', 2: 'force_z', 3: 'torque_x', 4: 'torque_y', 5: 'torque_z'},
    #           inplace=True)
    lengths = [len(sample_terrain[0]),  len(sample_metal[0])]
    samples = [[], [], [], [], [], []]
    for i in range(6):
        samples[i].extend(sample_terrain[i])
        samples[i].extend(sample_metal[i])
    # print(len(samples[0]) == len(sample_terrain[0]) + len(sample_metal[0]) + len(sample_concrete[0]))
    labels = []
    for i in range(len(sample_terrain[0])):
        labels.append(label_terrain)
    for i in range(len(sample_metal[0])):
        labels.append(label_metal)
    # print(labels)

    # sample_terrain = np.array(sample_terrain)
    # print(sample_terrain.shape)
    # sample_metal = np.array(sample_metal)
    # print(sample_metal.shape)
    # sample_concrete = np.array(sample_concrete)
    # print(sample_concrete.shape)

    # 转化为numpy
    samples = np.array(samples)
    # print(samples.shape)
    # print(samples[0])
    (h, w, t) = samples.shape  # (6, N, 200)
    sample_new = [[], [], [], [], [], []]
    for i in range(6):

        # Low Variance Filter   低方差滤波
        df = pd.DataFrame(samples[i])
        var = df.var()
        col = df.columns
        variable = []
        for j in range(0, len(var)):
            if var[j] >= 10:  # 将方差阈值设置为10
                # print()
                variable.append(col[j])
        if len(variable) <= 3:
            variable = []
            for k in range(0, len(var)):
                if var[k] >= 1:  # 将方差阈值设置为5
                    # print("Here")
                    variable.append(col[k])
        # print(len(variable))

        # Principle Variable Analysis  主成分分析
        df1 = df[variable]
        pca = decomposition.PCA(n_components=dim)  # n_components=0.98
        df1_np = np.array(df1)
        # print(df1_np.shape)
        sample_pca = pca.fit_transform(df1_np)
        # print(sample_pca.tolist())
        sample_new[i] = sample_pca.tolist()
        if dim < 1:
            print(len(pca.explained_variance_ratio_))
        # print(sample_new)
    # print(lengths)
    # print(len(sample_new[0]))
    moon_terrain, moon_metal = [[], [], [], [], [], []], [[], [], [], [], [], []]
    for i in range(6):
        moon_terrain[i] = sample_new[i][0:lengths[0]]
        moon_metal[i] = sample_new[i][lengths[0]:]
    label_terrain = labels[0:lengths[0]]
    label_metal = labels[lengths[0]:]
    print("Reduction to dim {0} finished".format(dim))

    return moon_terrain, moon_metal, label_terrain, label_metal

if __name__ == '__main__':
    moon_terrain, moon_metal, label_terrain, label_metal = dimReduction(dim = 3)
    # transfer to numpy format
    sample_terrain = np.array(moon_terrain)
    sample_metal = np.array(moon_metal)
    print(sample_metal.shape)