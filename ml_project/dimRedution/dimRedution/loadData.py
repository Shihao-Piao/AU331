"""
This file utilize the function of loading data.

@:param filepath: '../data/metal_data/force_x_data'
                '../data/concrete_data/force_x_data'
                '../data/result/force_x_data'
        mode : one, all28
@:return A list of data which indicate the same type.
"""

import os
import matplotlib.pyplot as plt


def load_data(filepath, mode):
    path = filepath
    data = []

    if mode == 'one':
        with open(path + '.txt', 'r') as f:
            line = f.readline()
            line = line[:-1]
            # cnt = 1
            while line:
                # print(cnt, line)
                data.append(float(line))
                line = f.readline()
                line = line[:-1]
                # cnt += 1
        f.close()
        # print("Load Data Finished.")
        return data
    elif mode == 'all28':
        for i in range(1, 29):
            with open(path + '_' + str(i) + '.txt', 'r') as f:
                line = f.readline()
                line = line[:-1]
                # cnt = 1
                while line:
                    # print(cnt, line)
                    data.append(float(line))
                    line = f.readline()
                    line = line[:-1]
            f.close()
        # print("Load Data Finished.")
        return data

def drawfig(data, mode):
    if mode == 'scatter':
        x = [i for i in range(len(a))]
        plt.plot(x, data)
        plt.show()
        print("Draw scatter figure finished.")


if __name__ == '__main__':
    a = load_data('../data/metal_data/force_x_data', 'one')
    plt.figure(1)
    drawfig(a, 'scatter')

    a = load_data('../data/result/force_x_data', 'all28')
    plt.figure(2)
    drawfig(a, 'scatter')

    a = load_data('../data/result/force_x_data_1', 'one')
    plt.figure(3)
    drawfig(a, 'scatter')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
