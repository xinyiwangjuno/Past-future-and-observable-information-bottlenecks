import matplotlib.pyplot as plt
import torch
from torch import nn, optim, autograd
from torch.nn import functional as F

import numpy as np
import scipy.spatial
import math
import os
import pyemma as pe
USE_GPU = torch.cuda.is_available()


Dx = 2
N =10000
kBT = 10
repeat_times =200
os.makedirs('./Data', exist_ok=True)

data_ori = np.loadtxt('Data/data_'+str(N)+'_'+str(kBT)+'_'+str(repeat_times)).reshape(-1,Dx+1)
print(np.shape(data_ori))

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(data_ori[:, 1], data_ori[:, 2], marker='o',lw = 0.75,c=data_ori[:,0], cmap = plt.cm.get_cmap('nipy_spectral', 20))
ax.set_title('V(x)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
fig.show()

fig, ax = pe.plots.plot_free_energy(data_ori[:,1],data_ori[:,2])
fig.show()

X0 = torch.from_numpy(data_ori[:1500000, 1:]).float()
print(X0.size())

X1 = torch.from_numpy(data_ori[100:1500100, 1:]).float()
print(X1.size())

train_batch_size = 128

train_loader = torch.utils.data.DataLoader(
    torch.cat([X0, X1], 1), batch_size = train_batch_size, shuffle = True)

Dr = 1

R_net = nn.Sequential(
    nn.Linear(Dx, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, Dr)
)

T_net = nn.Sequential(
    nn.Linear(Dx + Dr, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)


lr = 1e-4
optimizer = optim.Adam(list(R_net.parameters()) + list(T_net.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

n_epochs = 50
train_data_size = len(train_loader.dataset)
for epoch in range(1, n_epochs + 1):
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):

        actual_size = len(data)
        data = data
        data_X0 = data[:, :Dx]
        data_X1 = data[:, Dx:2 * Dx]
        # rand_ind = np.random.choice(train_data_size, actual_size, replace=False)
        # data_iX1 = train_loader.dataset[rand_ind, Dx:2*Dx].to(device)
        data_iX1 = data_X1[torch.randperm(actual_size)]
        if USE_GPU:
            data = data.cuda()
            data_X0 = data_X0.cuda()
            data_X1 = data_X1.cuda()
            data_iX1 = data_iX1.cuda()
        R0 = R_net(data_X0)
        T_J = T_net(torch.cat([R0, data_X1], 1))
        T_I = T_net(torch.cat([R0, data_iX1], 1))
        loss = -T_J.mean() + torch.logsumexp(T_I, 0)[0]

        train_loss += loss.item() * len(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)

    if epoch % 1 == 0:
        print('Train Epoch: {} ===> Train set loss: {:.4f}'.format(epoch, train_loss))

    scheduler.step()


# R = R_net(X0.to(device))
# plt.plot(X0_bar.cpu().detach().numpy()[:, 0], R.cpu().detach().numpy()[:, 0], 'o')
if USE_GPU:
    X0 = X0.cuda()


R = R_net(X0).detach().numpy()
fig, ax = plt.subplots(figsize=(8, 5))
x0 = X0[:, 0].detach().numpy()
x1 = X0[:, 1].detach().numpy()
ax2, fig2 = pe.plots.plot_free_energy(x0, x1)
ax2.show()
x0 = X0[:, 0].detach().numpy().reshape(-1, 1)
x1 = X0[:, 1].detach().numpy().reshape(-1, 1)
ax.scatter(x0, x1, c=R, marker='o', lw=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 20))
ax.set_title('R(X)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
fig.show()