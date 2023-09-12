"""
To cut data of each step
@:param data = 0 : metal 数据
        data = 1 : water 数据
        data = 2 : concrete 数据

string : path = 你储存数据的文件夹 如 path= '../data/metal_data/'

@:return 一个三维数组 sample 形状是 (6, N, 200)
         以及 label 是一个string 如： 'water', 'metal', 'concrete'
"""
from loadData import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# 每两百个是一步的数据，如 fx[0] = Fx[0: 200]
def cut_data(data = 0, path= '../data/metal_data/'):
    #
    if data == 0:
        Fx = load_data(path + 'force_x_data', 'one')
        Fy = load_data(path + 'force_y_data', 'one')
        Fz = load_data(path + 'force_z_data', 'one')
        Tx = load_data(path + 'torque_x_data', 'one')
        Ty = load_data(path + 'torque_y_data', 'one')
        Tz = load_data(path + 'torque_z_data', 'one')
        label = 'metal'
    elif data == 3:
        Fx = load_data(path + 'force_x_data', 'one')
        Fy = load_data(path + 'force_y_data', 'one')
        Fz = load_data(path + 'force_z_data', 'one')
        Tx = load_data(path + 'torque_x_data', 'one')
        Ty = load_data(path + 'torque_y_data', 'one')
        Tz = load_data(path + 'torque_z_data', 'one')
        label = 'moon_terrain'
    elif data == 4:
        Fx = load_data(path + 'force_x_data', 'one')
        Fy = load_data(path + 'force_y_data', 'one')
        Fz = load_data(path + 'force_z_data', 'one')
        Tx = load_data(path + 'torque_x_data', 'one')
        Ty = load_data(path + 'torque_y_data', 'one')
        Tz = load_data(path + 'torque_z_data', 'one')
        label = 'moon_metal'
    elif data == 2:
        Fx = load_data(path + 'force_x_data', 'one')
        Fy = load_data(path + 'force_y_data', 'one')
        Fz = load_data(path + 'force_z_data', 'one')
        # print(len(Fz))
        Tx = load_data(path + 'torque_x_data', 'one')
        Ty = load_data(path + 'torque_y_data', 'one')
        Tz = load_data(path + 'torque_z_data', 'one')
        label = 'concrete'
    elif data == 1:
        Fx = load_data(path + 'force_x_data', 'all28')
        Fy = load_data(path + 'force_y_data', 'all28')
        Fz = load_data(path + 'force_z_data', 'all28')
        Tx = load_data(path + 'torque_x_data','all28')
        Ty = load_data(path + 'torque_y_data','all28')
        Tz = load_data(path + 'torque_z_data','all28')
        label = 'water'
    idx = []  # 记录每个点对应的x坐标
    init_idx = []  # 记录每一步开始的x坐标
    fx, fy, fz, tx, ty, tz = [], [], [], [], [], []
    sample = [fx, fy, fz, tx, ty, tz]

    x = 0
    ther = 75
    if data == 4 or data == 3:
        ther = 22
    while x < len(Fx):
        if Fx[x] > ther :
            # print(x, Fx[x])
            cnt = 0
            if x + 200 > len(Fx):
                # fx.append(Fx[x:len(Fx)])
                # idx.append([i for i in range(x, len(Fx))])
                # init_idx.append(x)
                x = x + 200
                continue
            fx.append(Fx[x:x+200])
            idx.append([i for i in range(x, x+200)])
            init_idx.append(x)
            x = x + 200
        else:
            x += 1
    others = [Fy, Fz, Tx, Ty, Tz]
    _cnt = 1
    # print(init_idx)
    for other in others:
        for _idx in init_idx:
            # if _idx + 200 > len(other):
                # sample[_cnt].append(other[_idx:len(other)])
                # continue
            sample[_cnt].append(other[_idx:_idx + 200])
        _cnt += 1

        # if  plot == True or save == True:
        #     for i, a in enumerate(Fx):
        #         plt.plot(idx[i], a)
        #         plt.title("{0}_{1}-thKnock".format(lst[idx], i))
        #         if plot == True:
        #             plt.show()
        #         if save == True:
        #             plt.savefig("./timeDomainCut/{0}_{1}.jpg".format(lst[idx], i+1))
        #             plt.close()

    print('cut of data finished, ', len(fx), 'cutted')
    # print(len(sample[1]),len(sample[2]),len(sample[3]),len(sample[4]),len(sample[5]))

    return sample, label


if __name__ == '__main__':
    path = '../data/moon_metal_data/'  # metal 路径
    sample_metal, label_metal = cut_data(data=4, path=path)
    # b, c = cut_data(0, '../data/metal_data/')
    # b, c = cut_data(1, '../data/result/')
    # b, c = cut_data(2, '../data/concrete_data/')
    # print(len(b[2][400]))
    # # 转为dataframe
    # df = pd.DataFrame(b)
    # df = df.T
    # df.rename(columns={0: 'force_x', 1: 'force_y', 2:'force_z', 3:'torque_x', 4:'torque_y', 5:'torque_z'}, inplace=True)
    # print(df.shape)
    # # print(df)
