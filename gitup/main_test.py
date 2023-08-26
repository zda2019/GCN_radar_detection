import argparse
import os
import pickle as pk
from matplotlib import pyplot as plt
import numpy as np
import torch
from stgcn import STGCN
from matplotlib import pyplot as plt

torch.set_default_tensor_type(torch.FloatTensor)

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')

args = parser.parse_known_args()[0]
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

Num_of_train = 7
Num_of_test = 7
Start_num = 3
End_num = 3
Split_1 = 0.7
Split_2 = 0.9
Lr = 1e-3
epochs = 100
batch_size = 50

num_96=8


def generate_dataset(X, num_timesteps_input):
    indices = [(i, i + num_timesteps_input) for i in range(X[0].shape[2] - num_timesteps_input + 1)]
    print(X[0].shape)
    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(X[0][:, :, i: i + num_timesteps_input].transpose((0, 2, 1)))
        target.append(X[1][:, i + num_timesteps_input - 1])
    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target)).reshape(-1, X[0].shape[0], 1)


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


# 构建路径
path_all = []
for i in range(1, 16):
    S = "./save/dataset_t/result".__add__(str(i)).__add__(".mat")
    path_all.append(S)

# 训练数据构建
path = path_all[Num_of_train]
print(path)
str_t = path.split("/")[3].split(".")[0]

X = np.load("./save/dataset_t/" + str_t + "/node_feature.npy")[:, 0:-1, :]
Y = np.load("./save/dataset_t/" + str_t + "/node_feature.npy")[:, -1, :]
X = X.astype(np.float32)
Y = Y.astype(np.float32)
print(X.shape, Y.shape)

means = np.mean(X, axis=(0, 2))
X = X - means.reshape(1, -1, 1)
stds = np.std(X, axis=(0, 2))
X = X / stds.reshape(1, -1, 1)
split_line1 = int(X.shape[2] * Split_1)
split_line2 = int(X.shape[2] * Split_2)
#
train_original_data = (X[:, :, :split_line1], Y[:, :split_line1])
val_original_data = (X[:, :, split_line1:split_line2], Y[:, split_line1:split_line2])
test_original_data = (X[:, :, split_line2:], Y[:, split_line2:])
all_data = (X, Y)

num_timesteps_input = 20
training_input, training_target = generate_dataset(train_original_data,
                                                   num_timesteps_input=num_timesteps_input)
val_input, val_target = generate_dataset(val_original_data,
                                         num_timesteps_input=num_timesteps_input)
test_input, test_target = generate_dataset(test_original_data,
                                           num_timesteps_input=num_timesteps_input)

all_input, all_target = generate_dataset(all_data, num_timesteps_input=num_timesteps_input)

# for i in path_all[Start_num:End_num]:
#     path = i
#     print(path)
#     str_t = path.split("/")[3].split(".")[0]
#
#     X = np.load("./save/dataset_t/" + str_t + "/node_feature.npy")[:, 0:-1, :]
#     Y = np.load("./save/dataset_t/" + str_t + "/node_feature.npy")[:, -1, :]
#
#     X = X.astype(np.float32)
#     Y = Y.astype(np.float32)
#     print(X.shape, Y.shape)
#
#     means = np.mean(X, axis=(0, 2))
#     X = X - means.reshape(1, -1, 1)
#     stds = np.std(X, axis=(0, 2))
#     X = X / stds.reshape(1, -1, 1)
#
#     print(X.shape)
#
#     #     DataIndex=[i for i in range(X.shape[2])]
#     #     random.shuffle(DataIndex)
#     #     X=X[:,:,DataIndex]
#
#     #     print(X.shape)
#     split_line1 = int(X.shape[2] * 0.8)
#     split_line2 = int(X.shape[2] * 0.9)
#     #
#     train_original_data = (X[:, :, :split_line1], Y[:, :split_line1])
#     val_original_data = (X[:, :, split_line1:split_line2], Y[:, split_line1:split_line2])
#     test_original_data = (X[:, :, split_line2:], Y[:, split_line2:])
#
#     num_timesteps_input = 50
#     training_input_t, training_target_t = generate_dataset(train_original_data,
#                                                            num_timesteps_input=num_timesteps_input)
#     val_input_t, val_target_t = generate_dataset(val_original_data,
#                                                  num_timesteps_input=num_timesteps_input)
#     test_input_t, test_target_t = generate_dataset(test_original_data,
#                                                    num_timesteps_input=num_timesteps_input)
#
#     training_input = torch.cat((training_input, training_input_t), 0)
#     training_target = torch.cat((training_target, training_target_t), 0)
#     val_input = torch.cat((val_input, val_input_t), 0)
#     val_target = torch.cat((val_target, val_target_t), 0)
#     test_input = torch.cat((test_input, test_input_t), 0)
#     test_target = torch.cat((test_target, test_target_t), 0)

print(training_input.shape, training_target.shape)
print(val_input.shape, val_target.shape)
print(test_input.shape, test_target.shape)
print(all_input.shape, all_target.shape)

# 训练邻接矩阵
A = np.load("./save/dataset_t/adj_mat.npy")
print(A.shape)
A_wave = get_normalized_adj(A)
A_wave = torch.from_numpy(A_wave)

A_wave = A_wave.to(device=args.device)
A_wave = A_wave.float()

num_timesteps_output = 1
net = STGCN(A_wave.shape[0],
            training_input.shape[3],
            num_timesteps_input,
            num_timesteps_output).to(device=args.device)
print("A_wave", A_wave[0, 0], type(A_wave[0, 0]))
print('training_input', training_input[0, 0, 0, 0], type(training_input[0, 0, 0, 0]))
print("training_target", training_target[0, 0, 0], type(training_target[0, 0, 0]))

optimizer = torch.optim.Adam(net.parameters(), lr=Lr)

training_losses = []
validation_losses = []
validation_maes = []


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = torch.tensor(y_batch.to(device=args.device).view(-1).numpy().tolist(), dtype=torch.long)
        out = net(A_wave, X_batch).view(-1, 2)
        #         print("y_batch.shape,out.shape",y_batch.shape,out.shape)

        loss = net.cal_loss(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses) / len(epoch_training_losses)


for epoch in range(epochs):
    loss = train_epoch(training_input, training_target,
                       batch_size=batch_size)
    training_losses.append(loss)

    # Run validation
    with torch.no_grad():
        net.eval()
        val_input = val_input.to(device=args.device)
        val_target = val_target.to(device=args.device).view(-1).long()

        out = net(A_wave, val_input).view(-1, 2)
        print("val_target.shape,out.shape", val_target.shape, out.shape)
        val_loss = net.cal_loss(out, val_target)
        validation_losses.append(np.asscalar(val_loss.detach().numpy()))

        #         out_unnormalized = out.detach().cpu().numpy() * stds[0] + means[0]
        #         target_unnormalized = val_target.detach().cpu().numpy() * stds[0] + means[0]
        #         mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
        #         validation_maes.append(mae)

        out = None
        val_input = val_input.to(device="cpu")
        val_target = val_target.to(device="cpu")
    print("epoch:", epoch)
    print("Training loss: {}".format(training_losses[-1]))
    print("Validation loss: {}".format(validation_losses[-1]))

    checkpoint_path = "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open("checkpoints/losses.pk", "wb") as fd:
        pk.dump((training_losses, validation_losses, validation_maes), fd)

    plt.plot(training_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.legend()
    plt.show()

torch.save(net, './save/model_final.pkl')
net = torch.load("./save/model_final.pkl")
print(net)

# 训练数据恢复测试
# test.shape=395
out = net(A_wave, all_input)
out_unnormalized = out.detach().cpu().numpy() * stds[0] + means[0]
np.save("./save/all_out_PDF.npy", out_unnormalized[:, :, 1])

all_out = np.argmax(out_unnormalized, axis=2)
print(all_target.shape)
all_target = all_target.reshape(-1, all_target.shape[1]).numpy()

np.save("./save/all_out.npy", all_out)
np.save("./save/all_target.npy", all_target)

# io.savemat('./save/data_pred.mat', {'data_pred': all_out})
# io.savemat('./save/data_target.mat', {'data_target': all_target})


target = np.load("./save/all_out.npy")
out = np.load("./save/all_target.npy")
PDF = np.load("./save/all_out_PDF.npy")

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

fig3 = plt.figure()
plt.pcolor(PDF.T, cmap='jet')
plt.colorbar(shrink=.83)
plt.title("PDF")
plt.show()

# 泛化能力测试
path = path_all[Num_of_test]
str_t = path.split("/")[3].split(".")[0]
X = np.load("./save/dataset_t/" + str_t + "/node_feature.npy")[:, 0:-1, :]
Y = np.load("./save/dataset_t/" + str_t + "/node_feature.npy")[:, -1, :]
print(X.shape, Y.shape)
X = X.astype(np.float32)
Y = Y.astype(np.float32)
means = np.mean(X, axis=(0, 2))
X = X - means.reshape(1, -1, 1)
stds = np.std(X, axis=(0, 2))
X = X / stds.reshape(1, -1, 1)
all_data = (X, Y)
# num_timesteps_input = 50
all_input_t, all_target_t = generate_dataset(all_data, num_timesteps_input=num_timesteps_input)

A = np.load("./save/dataset_t/adj_mat.npy")
print(A.shape)
A_wave = get_normalized_adj(A)
A_wave = torch.from_numpy(A_wave)

A_wave = A_wave.to(device=args.device)
A_wave_t = A_wave.float()

net.eval()
out = net(A_wave_t, all_input_t)
out_unnormalized_t = out.detach().cpu().numpy() * stds[0] + means[0]
all_out_t = np.argmax(out_unnormalized_t, axis=2)
np.save("./save/all_out_PDF_t.npy", out_unnormalized[:, :, 1])
# print(all_target_t.shape)
all_target_t = all_target_t.reshape(-1, num_96).numpy()
# print(all_target_t.shape)
np.save("./save/all_out_t.npy", all_out_t)
np.save("./save/all_target_t.npy", all_target_t)

# io.savemat('./save/data_pred_t.mat', {'data_pred': all_out_t})
# io.savemat('./save/data_target_t.mat', {'data_target': all_target_t})

target = np.load("./save/all_out_t.npy")
out = np.load("./save/all_target_t.npy")
PDF = np.load("./save/all_out_PDF_t.npy")

fig4 = plt.figure()
plt.pcolor(target.T, cmap='jet')
plt.colorbar(shrink=.83)
plt.title("all_target_t")
plt.show()

fig5 = plt.figure()
plt.pcolor(out.T, cmap='jet')
plt.colorbar(shrink=.83)
plt.title("all_out_t")
plt.show()

fig6 = plt.figure()
plt.pcolor(PDF.T, cmap='jet')
plt.colorbar(shrink=.83)
plt.title("all_PDF_t")
plt.show()
