import torch

import numpy as np


def predicate(model_path, x: np.ndarray):
    # 载入模型
    model = torch.load(model_path)
    # 评估模式
    model.eval()
    # 格式对齐
    x = x.reshape(1, 5).astype(np.float32)

    # 预测
    predict = []
    for i in range(1, 32):
        x = torch.tensor(x).cuda()
        predict = model(x)  # 预测
        predict = predict.cpu().detach().numpy()  # 格式转换

    return predict
