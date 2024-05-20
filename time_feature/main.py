import os
import random

import joblib
import sys
import netron
import torch.onnx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

from model import *
from train import *
from test import *
from predict import *

# 环境载入
lstm_path = os.path.dirname(__file__)
data_path = os.path.join(lstm_path, "data")
output_path = os.path.join(lstm_path, "output")

# 归一化模型
train_x_scaler = MinMaxScaler()
train_y_scaler = MinMaxScaler()
test_x_scaler = MinMaxScaler()
test_y_scaler = MinMaxScaler()


def main(argv):
    # 参数设定
    batch_size = 1  # batch大小
    input_size = 5  # 输入层大小
    hidden_size = 200  # 隐藏层大小
    num_layers = 2  # LSTM堆叠层数
    output_size = 4  # 输出层大小
    epochs = 10  # 训练轮次
    learning_rate = 0.001  # 学习率
    seed = 4099416375  # 随机数种子
    save_pth_model_path = os.path.join(output_path, "model.pth")  # pth模型保存路径
    save_onnx_model_path = os.path.join(output_path, "model.onnx")  # onnx模型保存路径
    create_model_model = True  # 是否创建新模型
    train_mode = True  # 是否训练模型
    test_mode = True  # 是否测试模型
    convert_onnx_mode = True  # 是否转化为onnx模型

    # 数据加载
    data = []
    data_dirname_list = os.listdir(data_path)
    for data_dirname in data_dirname_list:
        data_single = pd.read_csv(os.path.join(data_path, data_dirname, "Data.csv"), header=0)
        data_single = np.array(data_single)  # 格式转换
        for i in range(data_single.shape[0] - 1):
            data.append(np.hstack((data_single[i, :5], data_single[i + 1, 1:5])))
    data = np.array(data)

    # 数据分类
    train_data = data[:int(data.shape[0] * 0.7)]
    valid_data = data[int(data.shape[0] * 0.7):int(data.shape[0] * 0.9)]
    test_data = data[int(data.shape[0] * 0.9):]

    # 归一化
    train_x_data = train_x_scaler.fit_transform(train_data[:, :5])
    train_y_data = train_y_scaler.fit_transform(train_data[:, 5:])
    valid_x_data = test_x_scaler.fit_transform(valid_data[:, :5])
    valid_y_data = test_y_scaler.fit_transform(valid_data[:, 5:])
    test_x_data = test_x_scaler.fit_transform(test_data[:, :5])
    test_y_data = test_y_scaler.fit_transform(test_data[:, 5:])

    # 创建数据集
    train_dataset = lstmDataset(train_x_data, train_y_data)
    valid_dataset = lstmDataset(valid_x_data, valid_y_data)
    test_dataset = lstmDataset(test_x_data, test_y_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    joblib.dump(train_x_scaler, os.path.join(output_path, "x_scaler.pkl"))
    joblib.dump(train_y_scaler, os.path.join(output_path, "y_scaler.pkl"))

    # 创建模型
    if create_model_model:
        torch.manual_seed(seed)
        # 写出随机数
        with open(os.path.join(output_path, "seed.txt"), 'w') as f:
            f.write(str(seed))
        model = LSTM(input_size, hidden_size, num_layers, output_size)
        torch.save(model, save_pth_model_path)

    # 训练模型
    if train_mode:
        train_loss, val_loss = train(save_pth_model_path, train_dataloader, valid_dataloader, epochs, learning_rate)
        # 保存训练轮次损失
        train_loss = pd.DataFrame(train_loss, index=None)
        train_loss.to_csv(os.path.join(output_path, "train_loss.csv"), index=False, header=False)
        val_loss = pd.DataFrame(val_loss, index=None)
        val_loss.to_csv(os.path.join(output_path, "val_loss.csv"), index=False, header=False)
        # 读入数据
        train_loss = pd.read_csv(os.path.join(output_path, "train_loss.csv"), header=None)
        val_loss = pd.read_csv(os.path.join(output_path, "val_loss.csv"), header=None)
        # 图像绘制
        plt.figure()
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MES Loss')
        plt.plot(train_loss, 'r', label='Train Loss')
        plt.plot(val_loss, 'b', label='Val Loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(output_path, "both_loss.png"))

    # 转换模型
    if convert_onnx_mode:
        model = torch.load(save_pth_model_path).cpu()
        model.eval()
        inputs = torch.randn(1, 5)
        torch.onnx.export(model, inputs, save_onnx_model_path)
        netron.start(save_onnx_model_path)  # 可视化

    # 测试模型
    if test_mode:
        test_loss = test(save_pth_model_path, test_dataloader)
        test_loss = pd.DataFrame(test_loss, index=None)
        test_loss.to_csv(os.path.join(output_path, "test_loss.csv"), index=False, header=False)

    # 预测模型
    x = np.array([0, 289.86500453948975, 3701.5, 17, 59])  # 默认测试参数
    if len(argv) == 5:  # 如果有输入参数则替换默认参数
        x[1:] = np.array(argv[1:])
    result = predicate(save_pth_model_path, x, train_x_scaler, train_y_scaler)
    print(result)  # 预测结果输出


# 数据集
class lstmDataset(Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, item):
        return self.data_x[item], self.data_y[item]

    def __len__(self):
        return self.data_x.shape[0]


if __name__ == '__main__':
    # 多次训练
    for index in range(1):
        # 创建文件夹
        output_path = os.path.join(lstm_path, "output", str(index))
        index += 1
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        main(sys.argv)
