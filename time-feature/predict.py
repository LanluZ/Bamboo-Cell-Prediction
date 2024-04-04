def predicate(test_x, model_path):  # 载入训练好的模型进行训练
    model = torch.load(model_path)
    model.eval()
    predict = []
    test_x = torch.tensor(test_x)
    with torch.no_grad():
        test_x = test_x.reshape(-1, 8)
        x1 = test_x.float().cuda()
        yy = model(x1)
        # yy=yy.squeeze()
        predict.append(yy.cpu().data.numpy())

    # predict = predict.reshape(-1, 1)
    # predict = mm_y.inverse_transform(predict)
    # plt.plot(range(len(predict)), predict)
    # plt.plot(range(len(predict)), y[10:, :])
    # plt.show()
    predict = torch.tensor(np.array(predict)).squeeze(1)
    predict = mm_y.inverse_transform(predict)
    return predict
