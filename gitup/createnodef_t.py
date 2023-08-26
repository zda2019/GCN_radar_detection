import os
import pickle
from math import sqrt
import numpy as np

import numpy
import scipy.io as scio
import math
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from datetime import datetime
import torch.optim as optim
from torch_geometric.data import DataLoader, InMemoryDataset, Data
from torch_geometric.datasets import Planetoid
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv  # noqa

e = math.e
pi = math.pi

path_all = []
for i in range(1, 16):
    S = "./CSIR/resultF2/result".__add__(str(i)).__add__(".mat")
    path_all.append(S)

# # 路径
# path_1 = "./CSIR/resultF2/result1.mat"
# path_4 = "./CSIR/resultF2/result4.mat"
# path_5 = "./CSIR/resultF2/result5.mat"
for i in path_all:
    print(i)
    path = i
    str = path.split("/")[3].split(".")[0]
    if not os.path.exists("./save/dataset_t/" + str + "/"):
        # os.mkdir创建一个，os.makedirs可以创建路径上多个
        os.makedirs("./save/dataset_t/" + str + "/")
num_timesteps_input = 20


# DFT模块
# 输入为1*n的离散序列，N为输出位数
# 输出为长度为N的DFT结果
def dft_f(xn, N):
    Xk = []
    l = len(xn)
    if l != N:
        for i in range(N - l):
            xn.append(0)
    w = e ** (-1j * 2 * pi / N)
    for i in range(N):
        m_index = 0
        for j in range(N):
            m_index = m_index + xn[j] * (w ** (i * j))
        Xk.append(m_index)
    return Xk


# 虚数转实数函数
# 输入为虚数序列，输出为对应的float数列
def xushu2float(arr):
    for i in range(len(arr)):
        arr[i] = abs(arr[i])
    return arr


arr_all_save = []


def create_t(path):
    print("data processing......")
    mydata = scio.loadmat(path)
    global arr_all_save

    # 时间窗分帧
    times = int(len(mydata['Time']) / 200)
    print(times)

    # 构造arr_all:(3393, 96, 204)
    arr_all = []
    for i in range(times):
        print(i)
        # 计算窗下GPS_Range
        length = 0
        for m in range(i * 200, (i + 1) * 200):
            length = length + (mydata["arr"])[m, 0]
        length_obj = length / 200

        # 构造arr_range:(96*1*203)
        arr_range = []
        for k in range(0, 96):

            # 构建距离维数据
            # arr:(204的数组)，前200维为特征值，204维为节点编号
            arr = []
            for j in range(i * 200, (i + 1) * 200):
                arr.append((mydata["X"])[j, k])

            if abs(length_obj - mydata['Range'][0, k]) <= 7.5:
                arr.append(1)  # 1表示有目标
            else:
                arr.append(0)  # 0表示无目标

            arr_range.append(arr)
        arr_all.append(arr_range)
    arr_all_save = arr_all
    return arr_all_save

path='./CSIR/resultF2/result8.mat'
create_t(path)
print(numpy.array(arr_all_save).shape)  # (566, 395, 201)

X = (numpy.array(arr_all_save)).transpose((1, 2, 0))  # ( 395, 201, 566)
print(X[:, :, 1] == (numpy.array(arr_all_save))[1, :, :])


np.save("./save/dataset_t/result8/node_feature.npy", X)
X = np.load("./save/dataset_t/result8/node_feature.npy")

# np.save("./save/dataset_t/" + str + "/node_feature.npy", X)
# X = np.load("./save/dataset_t/" + str + "/node_feature.npy")

#
