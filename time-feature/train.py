import torch
import numpy as np
import torch.nn as nn


# 训练模型
def train(model_path: str, dataloader, epochs: int, learn_rate: float):
    # 载入模型
    model = torch.load(model_path).cuda()
    # 训练模式
    model.train()

    # 超参数设置
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    # 训练轮次
    epoch_losses = []  # 每轮平均损失记录
    for epoch in range(epochs):
        # 轮次
        losses = []  # 本轮损失记录
        for i, (inputs, labels) in enumerate(dataloader):
            # 加载数据到GPU
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()

            optimizer.zero_grad()  # 梯度清零
            pred = model(inputs)  # 前向传播
            loss = loss_function(pred, labels)  # 损失计算
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新

            # 信息输出
            losses.append(loss.cpu().detach().numpy())

        # 信息输出
        losses_mean = np.mean(losses)
        epoch_losses.append(losses_mean)
        print("训练轮次 {} : 平均损失 {}".format(epoch, losses_mean))

    # 返回损失
    return epoch_losses
