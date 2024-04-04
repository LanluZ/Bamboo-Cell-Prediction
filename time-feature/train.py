def training(n_epoch, learn_rate, train, model):  # (训练次数，学习率，训练集，模型)
    model.train()
    loss_f = nn.MSELoss()  # MSE损失函数
    t_batch = len(train)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)  # 选取优化器
    total_loss, best_loss = 0, 10000000
    for epoch in range(n_epoch):
        total_loss = 0
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.float().cuda()  # 将数据放入GPU
            labels = labels.float().cuda()
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)
            # outputs=outputs.squeeze()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (t_batch * 50)
        print("avg_loss:", avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model, os.path.join(cwd_path, "ckpt.model"))
            print("saving model")