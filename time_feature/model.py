import torch.nn as nn


class LSTM(nn.Module):
    # 初始化
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(nn.Linear(hidden_size, output_size))

    # 前向传播
    def forward(self, x):
        x, _ = self.lstm(x, None)
        x = self.classifier(x)
        return x
