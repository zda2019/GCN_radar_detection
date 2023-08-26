from matplotlib import pyplot as plt
import numpy as np

target = np.load("./save/all_target.npy")
out = np.load("./save/all_out.npy")

fig1 = plt.figure()
plt.pcolor(target.T, cmap='jet')
plt.colorbar(shrink=.83)
plt.title("all_target")
plt.show()



fig2 = plt.figure()
plt.pcolor(out.T, cmap='jet')
plt.colorbar(shrink=.83)
plt.title("all_out")
plt.show()

