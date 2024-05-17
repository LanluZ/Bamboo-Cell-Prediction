import torch

import numpy as np


def predicate(model_path, x: np.ndarray, scaler_x, scaler_y):
    # 载入模型
    model = torch.load(model_path)
    # 评估模式
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    # 格式对齐
    x = x.reshape(1, 5)
    x = torch.tensor(x)

    # 预测
    result = []
    for i in range(32):
        x[0, 0] = i  # 序号添加
        result.append(x.cpu().numpy())  # 记录结果
        x = torch.tensor(scaler_x.transform(x.cpu())).type(torch.float32).cuda()
        y = scaler_y.inverse_transform(model(x).cpu().detach())
        x[:, 1:] = torch.tensor(y).cuda()

    result = np.array(result).squeeze()

    return result
