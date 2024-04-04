import torch

import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size  # 输入特征维度（label也作为一项特征）
        self.hidden_size = hidden_size  # 隐藏层参数
        self.num_layers = num_layers  # LSTM 堆叠的层数，默认值是1层，如果设置为2，第二个LSTM接收第一个LSTM的计算结果
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 2, 7))

    def forward(self, x):
        x, _ = self.lstm(x, None)

        x = self.calssifier(x)
        return x
