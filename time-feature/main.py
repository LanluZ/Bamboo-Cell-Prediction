import os
import netron
import torch.onnx

import numpy as np
import pandas as pd

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
train_scaler = MinMaxScaler()
test_scaler = MinMaxScaler()


def main():
    # 参数设定
    batch_size = 1  # batch大小
    input_size = 5  # 输入层大小
    hidden_size = 200  # 隐藏层大小
    num_layers = 2  # LSTM堆叠层数
    output_size = 4  # 输出层大小
    epochs = 10  # 训练轮次
    learning_rate = 0.001  # 学习率
    save_pth_model_path = os.path.join(output_path, "model.pth")  # pth模型保存路径
    save_onnx_model_path = os.path.join(output_path, "model.onnx")  # onnx模型保存路径
    create_model_model = False  # 是否创建新模型
    train_mode = False  # 是否训练模型
    test_mode = False  # 是否测试模型
    convert_onnx_mode = False  # 是否转化为onnx模型

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
    test_data = data[int(data.shape[0] * 0.7):]

    # 归一化
    train_data = train_scaler.fit_transform(train_data)
    test_data = test_scaler.fit_transform(test_data)

    # 创建数据集
    train_dataset = lstmDataset(train_data)
    test_dataset = lstmDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 创建模型
    if create_model_model:
        model = LSTM(input_size, hidden_size, num_layers, output_size)
        torch.save(model, save_pth_model_path)

    # 训练模型
    if train_mode:
        train_loss = train(save_pth_model_path, train_dataloader, epochs, learning_rate)
        # 保存训练轮次损失
        train_loss = pd.DataFrame(train_loss, index=None)
        train_loss.to_csv(os.path.join(output_path, "train_loss.csv"), index=False, header=False)

    # 转换模型
    if convert_onnx_mode:
        model = torch.load(save_pth_model_path).cpu()
        model.eval()
        inputs = torch.randn(1, 5)
        torch.onnx.export(model, inputs, save_onnx_model_path)
        netron.start(save_onnx_model_path)  # 可视化

    # 测试模型
    if test_mode:
        test(save_pth_model_path, test_dataloader)

    # 预测模型
    x = np.array([0, 289.86500453948975, 3701.5, 17, 59])
    result = predicate(save_pth_model_path, x)
    print(result)


# 数据集
class lstmDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data_x = data[:, :5]
        self.data_y = data[:, 5:]

    def __getitem__(self, item):
        return self.data_x[item], self.data_y[item]

    def __len__(self):
        return self.data_x.shape[0]


if __name__ == '__main__':
    main()
