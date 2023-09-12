"""
对数据进行降维处理和绘图可视化
@:param dim: 降到几维 1, 2, ...

在dimReduction函数里改 path（前六行） = 你储存数据的文件夹 如 path= '../data/metal_data/'

@:return 一个降维后的dim维数组 sample 形状是 (6, N, dim)
         以及 label 是一个数组,对应sample数组中样本的标签 如： [ 'metal',..., 'concrete',..., 'water','water',...,]
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

def dimReduction(dim = 3.): ## dim means the dimention to be reduced
    path = '../data/metal_data/'   # metal 路径
    sample_metal, label_metal = cut_data(data=0, path=path)
    path = '../data/result/'        # water 路径
    sample_water, label_water = cut_data(data=1, path=path)
    path = '../data/concrete_data/'  # concrete 路径
    sample_concrete, label_concrete = cut_data(data=2, path=path)
    ## 转化为dataframe
    # df = pd.DataFrame(sample)
    # df = df.T
    # df.rename(columns={0: 'force_x', 1: 'force_y', 2: 'force_z', 3: 'torque_x', 4: 'torque_y', 5: 'torque_z'},
    #           inplace=True)
    lengths = [len(sample_metal[0]), len(sample_concrete[0]), len(sample_water[0])]
    samples = [[],[],[],[],[],[]]
    for i in range(6):
        samples[i].extend(sample_metal[i])
        samples[i].extend(sample_concrete[i])
        samples[i].extend(sample_water[i])
    # print(len(samples[0]) == len(sample_metal[0]) + len(sample_water[0]) + len(sample_concrete[0]))
    labels = []
    for i in range(len(sample_metal[0])):
        labels.append(label_metal)
    for i in range(len(sample_concrete[0])):
        labels.append(label_concrete)
    for i in range(len(sample_water[0])):
        labels.append(label_water)
    # print(labels)

    # sample_metal = np.array(sample_metal)
    # print(sample_metal.shape)
    # sample_water = np.array(sample_water)
    # print(sample_water.shape)
    # sample_concrete = np.array(sample_concrete)
    # print(sample_concrete.shape)

    # 转化为numpy
    samples = np.array(samples)
    # print(samples.shape)
    # print(samples[0])
    (h, w, t) = samples.shape   # (6, N, 200)
    sample_new = [[],[],[],[],[],[]]
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
        pca = decomposition.PCA(n_components=dim) # n_components=0.98
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
    earth_metal, earth_concrete,earth_water = [[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]
    for i in range(6):
        earth_metal[i] = sample_new[i][0:lengths[0]]
        earth_concrete[i] = sample_new[i][lengths[0]:lengths[0]+lengths[1]]
        earth_water[i] = sample_new[i][lengths[0]+lengths[1]:]
    label_metal = labels[0:lengths[0]]
    label_concrete = labels[lengths[0]:lengths[0]+lengths[1]]
    label_water = labels[lengths[0]+lengths[1]:]
    print("Reduction to dim {0} finished".format(dim))
    
    return earth_metal, earth_concrete, earth_water, label_metal, label_concrete, label_water

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
    # 参数里面的3可以改成6 变成降到6维
    earth_metal, earth_concrete, earth_water, label_metal, label_concrete, label_water = dimReduction(dim=3)
    # print(earth_water)
    # earth_metal = np.array(earth_metal)

    # sample_dim2, _ = dimReduction(dim=2)
    # print(labels)
    # 绘制3D图像
    # x1 = [item[0] for item in earth_metal[0]]
    # y1 = [item[1] for item in earth_metal[0]]
    # z1 = [item[2] for item in earth_metal[0]]
    # x2 = [item[0] for item in earth_water[0]]
    # y2 = [item[1] for item in earth_water[0]]
    # z2 = [item[2] for item in earth_water[0]]
    # x3 = [item[0] for item in earth_concrete[0]]
    # y3 = [item[1] for item in earth_concrete[0]]
    # z3 = [item[2] for item in earth_concrete[0]]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(y1, x1, z1, c='r', label='metal')
    # ax.scatter(y2, x2, z2, c='b', label='water')
    # ax.scatter(y3, x3, z3, c='g', label='concrete')
    # ax.legend()
    # ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
    # plt.title("3D PCA Distribution")
    # plt.show()