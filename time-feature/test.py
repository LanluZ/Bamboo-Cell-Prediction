import torch


def test(model_path: str, dataloader):
    # 载入模型
    model = torch.load(model_path).cuda()
    # 测试模式
    model.eval()


