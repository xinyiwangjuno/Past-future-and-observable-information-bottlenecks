import torch
import numpy as np
import os

USE_GPU = torch.cuda.is_available()

Dx = 2
N = 10000
kBT = 10
repeat_times = 200
os.makedirs('./Data', exist_ok=True)

aa = [-1, -1, -6.5, 0.7]
bb = [0, 0, 11, 0.6]
cc = [-10, -10, -6.5, 0.7]
AA = [-200, -100, -170, 15]
XX = [1, 0, -0.5, -1]
YY = [0, 0.5, 1.5, 1]
sigma = 0.05


def get_V(px):
    px = np.array(px)
    ee = 0
    if np.size(px.shape) == 2:
        for j in range(4):
            ee = ee + AA[j] * np.exp(aa[j] * (px[:, 0] - XX[j]) ** 2 +
                                     bb[j] * (px[:, 0] - XX[j]) * (px[:, 1] - YY[j]) +
                                     cc[j] * (px[:, 1] - YY[j]) ** 2)
        ee += 9 * np.sin(2 * 5 * np.pi * px[:, 0]) * np.sin(2 * 5 * np.pi * px[:, 1])
        for i in range(2, Dx):
            ee += px[:, i] ** 2 / 2 / sigma ** 2
    else:
        for j in range(4):
            ee = ee + AA[j] * np.exp(aa[j] * (px[0] - XX[j]) ** 2 +
                                     bb[j] * (px[0] - XX[j]) * (px[1] - YY[j]) +
                                     cc[j] * (px[1] - YY[j]) ** 2)
        ee += 9 * np.sin(2 * 5 * np.pi * px[0]) * np.sin(2 * 5 * np.pi * px[1])
        for i in range(2, Dx):
            ee += px[i] ** 2 / 2 / sigma ** 2
    return ee


def get_grad(px):
    px = np.array(px)
    gg = np.zeros(shape=(Dx,), dtype=np.float64)

    for j in range(4):
        ee = AA[j] * np.exp(aa[j] * (px[0] - XX[j]) ** 2 +
                            bb[j] * (px[0] - XX[j]) * (px[1] - YY[j]) +
                            cc[j] * (px[1] - YY[j]) ** 2)
        gg[0] = gg[0] + (2 * aa[j] * (px[0] - XX[j]) +
                         bb[j] * (px[1] - YY[j])) * ee
        gg[1] = gg[1] + (bb[j] * (px[0] - XX[j]) +
                         2 * cc[j] * (px[1] - YY[j])) * ee
    gg[0] += 9 * 2 * 5 * np.pi * np.cos(2 * 5 * np.pi * px[0]) * np.sin(2 * 5 * np.pi * px[1])
    gg[1] += 9 * 2 * 5 * np.pi * np.sin(2 * 5 * np.pi * px[0]) * np.cos(2 * 5 * np.pi * px[1])
    for i in range(2, Dx):
        gg[i] = px[i] / sigma ** 2
    return gg


def sim_data(px_init, kBT, dt, D_size, firstsave=1000, t_sep=100):
    D = []
    px = px_init
    id_ = 0
    i = 0
    while True:
        px = px - dt * get_grad(px) + np.sqrt(2 * kBT * dt) * np.random.normal(size=(Dx,))
        if i >= firstsave and i % t_sep == 0 and 1 >= px[0] >= -1.5 \
                and 2 >= px[1] >= -0.5:
            D.append(px)
            id_ += 1
            if id_ >= D_size:
                break
        i += 1
    data = np.zeros((D_size, 1 + Dx))
    data[:, 1:] = np.array(D)
    data[:, 0] = get_V(D)
    #     print('generating data finished!')
    return data


data_ori = []
for i in range(repeat_times):
    x1 = np.random.uniform(low=-1.0, high=1.0)
    x2 = np.random.uniform(low=-1.0, high=1.0)
    px_init = np.zeros((Dx,))
    px_init[0] = x1
    px_init[1] = x2
    data_ori_t = sim_data(px_init=px_init, kBT=kBT, dt=1e-5, D_size=N)
    data_ori.extend(data_ori_t)
print(np.shape(data_ori))


np.savetxt('Data/data_'+str(N)+'_'+str(kBT)+'_'+str(repeat_times)+'_2', data_ori)
