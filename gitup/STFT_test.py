from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

dataset_1 = scio.loadmat("./CSIR/resultF2/result1.mat")["X"]
print(dataset_1.shape)
for i in range(96):
    print(i)
    STFT_data_input = abs(dataset_1[:, i].reshape(-1))
    print(STFT_data_input.shape)
    fs = 1e+5
    window = 'hann'
    n = 120

    # f, t, Z = stft(STFT_data_input, fs=fs, window=window, nperseg=n)
    f, t, Z = stft(STFT_data_input, window=window, nperseg=480)
    print(Z)
    print(Z.shape)
    Z = np.abs(Z)
    # 如下图所示
    plt.pcolormesh(t, f, Z, vmin=0, vmax=Z.mean() * 10)
    plt.title("arr-" + str(i))
    plt.show()
