import os
import sys

import netron
import torch

import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from model import *
from train import *
from test import *

# 环境载入
dnn_path = os.path.dirname(__file__)
data_path = os.path.join(dnn_path, "data")
output_path = os.path.join(dnn_path, "output")

# 归一化模型
train_x_scaler = MinMaxScaler()
train_y_scaler = MinMaxScaler()
test_x_scaler = MinMaxScaler()
test_y_scaler = MinMaxScaler()


def main(argv):
    # 参数设定
    input_size = 7  # 输入层大小
    hidden_size = 4  # 隐藏层大小
    output_size = 1  # 输出层大小
    epochs = 20  # 训练轮次
    batch_size = 4  # 批次大小
    learning_rate = 0.002  # 学习率
    save_pth_model_path = os.path.join(output_path, "model.pth")  # pth模型保存路径
    save_onnx_model_path = os.path.join(output_path, "model.onnx")  # onnx模型保存路径
    create_model_model = False  # 是否创建新模型
    train_mode = False  # 是否训练模型
    test_mode = False  # 是否测试模型
    convert_onnx_mode = False  # 是否转化为onnx模型

    # 数据加载
    data = pd.read_csv(os.path.join(dnn_path, "data", "snum.csv"), header=0)
    data = np.array(data.iloc[:, 1:])

    # 数据分类
    train_data = data[:int(data.shape[0] * 0.7)]
    test_data = data[int(data.shape[0] * 0.7):]

    train_x_data = train_data[:, :-1]
    train_y_data = train_data[:, -1:]
    test_x_data = test_data[:, :-1]
    test_y_data = test_data[:, -1:]

    # 归一化
    train_x_data = train_x_scaler.fit_transform(train_x_data)
    train_y_data = train_y_scaler.fit_transform(train_y_data)
    test_x_data = test_x_scaler.fit_transform(test_x_data)
    test_y_data = test_y_scaler.fit_transform(test_y_data)

    # 创建数据集
    train_dataset = dnnDataset(train_x_data, train_y_data)
    test_dataset = dnnDataset(test_x_data, test_y_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # 模型创建
    if create_model_model:
        model = DNN(input_size, hidden_size, output_size)
        torch.save(model, save_pth_model_path)

    # 训练模型
    if train_mode:
        train_loss = train(save_pth_model_path, train_dataloader, epochs, learning_rate)
        train_loss = pd.DataFrame(train_loss, index=None)
        train_loss.to_csv(os.path.join(output_path, "train_loss.csv"), index=False, header=False)

    # 测试模型
    if test_mode:
        test_loss = test(save_pth_model_path, test_dataloader)
        test_loss = pd.DataFrame(test_loss, index=None)
        test_loss.to_csv(os.path.join(output_path, "test_loss.csv"), index=False, header=False)

    # 转化模型
    if convert_onnx_mode:
        model = torch.load(save_pth_model_path)
        inputs = torch.randn(1, 7).cuda()
        torch.onnx.export(model, inputs, save_onnx_model_path)  # 模型转化
        netron.start(save_onnx_model_path)

    # 测试模型
    if test_mode:
        test_loss = test(save_pth_model_path, test_dataloader)
        test_loss = pd.DataFrame(test_loss, index=None)
        test_loss.to_csv(os.path.join(output_path, "test_loss.csv"), index=False, header=False)

    # 预测模型
    x = np.array(
        [56.28427052497864, 219.0, 7.0, 9.0, 0.7777777777777778, 0.7684210526315789, 0.8687201504405094])  # 默认测试参数
    if len(argv) == 8:  # 如果有输入参数则替换默认参数
        x = np.array(argv[1:])
    model = torch.load(save_pth_model_path)
    result = model(torch.tensor(train_x_scaler.transform(x.reshape((1, 7)))).float().cuda())
    result = train_y_scaler.inverse_transform(result.cpu().detach().numpy())
    print(result)  # 预测结果输出


class dnnDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, item):
        return self.data_x[item], self.data_y[item]

    def __len__(self):
        return self.data_x.shape[0]


if __name__ == '__main__':
    main(sys.argv)
