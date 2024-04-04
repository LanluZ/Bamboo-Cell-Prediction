import os
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

cwd_path = os.getcwd()
path = os.path.join(cwd_path, "实验数据")  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
mm_x = MinMaxScaler()
mm_y = MinMaxScaler()

print(files)


def load_data(path_x):  # 通过列表读取
    x = pd.read_csv(path_x, header=0).values
    return x


def creat_XY(datasets, n_past=1):  # n_past 为滑动窗口，通过前n_past个特征加上label 预测下一个label
    dataX = []
    dataY = []
    for dataset in datasets:
        for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i, 1:])
    dataX = torch.tensor(np.array(dataX))
    dataY = torch.tensor(np.array(dataY))
    dataX = dataX.squeeze(1)
    data_normal_x = mm_x.fit_transform(dataX)
    data_normal_y = mm_y.fit_transform(dataY)
    return data_normal_x, data_normal_y


class LSTM_Dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, item):
        return self.data_x[item], self.data_y[item]

    def __len__(self):
        return len(self.data_y)


if __name__ == '__main__':
    train_data = []
    for file in files:
        try:
            temp = load_data(path + "\\" + file + "\\Data.csv")
            temp.tolist()
            train_data.append(temp)
        except FileNotFoundError:
            pass

    n_past = 1  # 滑动窗口
    batch_size = 50  # 表示单次传递给程序用以训练的数据（样本）个数
    lr = 0.0002  # 学习率

    # 根据滑动窗口大小划分数据集
    data_x, data_y = creat_XY(train_data, n_past)

    print(data_y.shape)
    lstm = LSTM(input_size=data_x.shape[-1], hidden_size=350, num_layers=2).cuda()
    dataset = LSTM_Dataset(data_x, data_y)
    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    # training(n_epoch=1000,lr=lr,train=train_loader,model=lstm)
    # predicate(train_loader,"ckpt.model")
    predict = predicate(data_x[25], os.path.join(cwd_path, "ckpt.model"))
    # predict = mm_y.inverse_transform(zz)
    predict = list(predict.squeeze(0))
    print("预测结果", predict)

    # 绘图
    plt.plot(range(len(predict)), predict)
    plt.plot(range(len(predict)), train_data[0][1][1:])
    plt.legend(['predict', 'real'])  # 打出图例
    plt.show()
