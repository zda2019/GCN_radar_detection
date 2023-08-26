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

# 路径
path_1 = "./CSIR/resultF2/result1.mat"
path_4 = "./CSIR/resultF2/result4.mat"
path_5 = "./CSIR/resultF2/result5.mat"

path = path_5
str = path.split("/")[3].split(".")[0]
if not os.path.exists("./save/dataset/" + str + "/"):
    # os.mkdir创建一个，os.makedirs可以创建路径上多个
    os.makedirs("./save/dataset/" + str + "/")

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


def create(path):
    mydata = scio.loadmat(path)
    global arr_all_save

    times = int(len(mydata['Time']) / 200)
    print(times)

    arr_all = []
    m_index = 0
    for i in range(times):
        print(i)

        # 计算窗下GPS_Range
        length = 0
        for m in range(i * 200, (i + 1) * 200):
            length = length + (mydata["arr"])[m, 0]
        length_obj = length / 200

        # 构造arr_range:
        arr_range = []
        for k in range(96):
            # 构建距离维数据
            # arr:(204的数组)，前200维为特征值，201维为节点目标特征
            if abs(length_obj - mydata['Range'][0, k]) <= 7.5:  #
                # print("object   ")
                for t in range(4):
                    # 节点重复:node1
                    for q in range(k - 10, k):
                        arr = []
                        for j in range(i * 200, (i + 1) * 200):
                            arr.append((mydata["X"])[j, q])
                        arr = dft_f(arr, 200)
                        arr = xushu2float(arr)
                        arr.append(0)
                        arr_range.append(arr)
                    # 节点重复:node2
                    arr = []
                    for j in range(i * 200, (i + 1) * 200):
                        arr.append((mydata["X"])[j, k])
                    arr = dft_f(arr, 200)
                    arr = xushu2float(arr)
                    arr.append(1)
                    arr_range.append(arr)
                    # 节点重复:node3
                    for q in range(k + 1, k + 11):
                        arr = []
                        for j in range(i * 200, (i + 1) * 200):
                            arr.append((mydata["X"])[j, k + 1])
                        arr = dft_f(arr, 200)
                        arr = xushu2float(arr)
                        arr.append(0)
                        arr_range.append(arr)
            else:
                arr = []
                for j in range(i * 200, (i + 1) * 200):
                    arr.append((mydata["X"])[j, k])
                arr = dft_f(arr, 200)
                arr = xushu2float(arr)
                arr.append(0)  # 0表示无目标
                arr_range.append(arr)
        arr_all.append(arr_range)
    arr_all_save = arr_all
    return arr_all


create(path)
print(numpy.array(arr_all_save).shape)  # (566, 395, 201)

X = (numpy.array(arr_all_save)).transpose((1, 2, 0))  # ( 395, 201, 566)

print(X[:, :, 1] == (numpy.array(arr_all_save))[1, :, :])

np.save("./save/dataset/" + str + "/node_feature.npy", X)
X = np.load("./save/dataset/" + str + "/node_feature.npy")
