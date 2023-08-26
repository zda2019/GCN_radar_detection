import numpy as np


def createAdj(length):
    arr_all = []
    for i in range(length):
        arr = []
        if i == 0:
            for j in range(2):
                arr.append(1.0 / (abs(i - j) + 1))
            for k in range(i + 2, length):
                arr.append(0)
        elif i == 1:
            for j in range(3):
                arr.append(1.0 / (abs(i - j) + 1))
            for k in range(3, length):
                arr.append(0)
        elif i == length - 1:
            for k in range(length - 2):
                arr.append(0)
            for j in range(length - 2, length):
                arr.append(1.0 / (abs(i - j) + 1))
        elif i == length - 2:
            for k in range(length - 3):
                arr.append(0)
            for j in range(length - 3, length):
                arr.append(1.0 / (abs(i - j) + 1))
        else:
            for k in range(i - 1):
                arr.append(0)
            for j in range(i - 1, i + 2):
                arr.append(1.0 / (abs(i - j) + 1))
            for m in range(i + 2, length):
                arr.append(0)
        arr_all.append(arr)

    return np.array(arr_all)


A = createAdj(395)

np.save("./save/dataset/adj_mat.npy", A)
A = np.load("./save/dataset/adj_mat.npy")
print(A.shape)
print(A)
