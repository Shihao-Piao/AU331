"""
对数据进行降维处理和绘图可视化
@:param data = 0 : metal 数据
        data = 1 : water 数据
        data = 2 : concrete 数据

string : path = 你储存数据的文件夹 如 path= '../data/metal_data/'

@:return 一个降维后的三维数组 sample 形状是 (6, N, 3)
         以及 label 是一个string 如： 'water', 'metal', 'concrete'
"""

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

def dimReduction(data = 0, path= '../data/metal_data/'): ## data in {0, 1}
    sample, label = cut_data(data=data, path=path)
    ## 转化为dataframe
    # df = pd.DataFrame(sample)
    # df = df.T
    # df.rename(columns={0: 'force_x', 1: 'force_y', 2: 'force_z', 3: 'torque_x', 4: 'torque_y', 5: 'torque_z'},
    #           inplace=True)

    # 转化为numpy
    sample = np.array(sample)
    # print(sample.shape)
    (h, w, t) = sample.shape   # (6, N, 200)
    sample_new = [[],[],[],[],[],[]]
    for i in range(6):

        # Low Variance Filter   低方差滤波
        df = pd.DataFrame(sample[i])
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
        pca = decomposition.PCA(n_components=3) # n_components=0.98
        df1_np = np.array(df1)
        # print(df1_np.shape)
        sample_pca = pca.fit_transform(df1_np)
        # print(sample_pca.tolist())
        sample_new[i] = sample_pca.tolist()
        # print(pca.explained_variance_ratio_)
        # print(sample_new)
    return sample_new, label
    # print(df1.corr())
    # print(fft_notonehot[:1])
    # 随机森林降维
    # df1 = df1.drop(['index'], axis=1)
    # model = RandomForestRegressor(random_state=1, max_depth=10)
    # df = pd.get_dummies(df1)
    # model.fit(df, df1.)

    # random.shuffle(fft)
    # fft_sample = [iter[0] for iter in fft[0: int(0.8 * len(fft))]]
    # fft_lable = [iter[1] for iter in fft[0: int(0.8 * len(fft))]]
    # test_sample = [iter[0] for iter in fft[int(0.8 * len(fft)):]]
    # test_lable = [iter[1] for iter in fft[int(0.8 * len(fft)):]]
    # return fft_sample, fft_lable, test_sample, test_lable


if __name__ == '__main__':
    sample_metal, label = dimReduction(0, '../data/metal_data/')
    sample_metal = np.array(sample_metal)
    print(sample_metal.shape, label)

    sample_water, label = dimReduction(1, '../data/result/')
    sample_water = np.array(sample_water)
    print(sample_water.shape, label)

    sample_concrete, label = dimReduction(2, '../data/concrete_data/')
    sample_concrete = np.array(sample_concrete)
    print(sample_concrete.shape, label)
