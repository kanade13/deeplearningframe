class Config:
    # 模型超参数
    hidden_size1 = 128  # 隐藏层大小
    hidden_size2 = 64  # 隐藏层大小
    output_size = 1   # 输出层大小（对于二分类任务）
<<<<<<< HEAD
    learning_rate = 0.01  # 学习率
    num_epochs = 10  # 训练轮数
=======
    learning_rate = 0.001  # 学习率
    num_epochs = 20  # 训练轮数
>>>>>>> f19c3d304a4fca624633d6614ecd82899e8a2ae7
    batch_size = 32   # 批次大小

    # 数据相关参数
    input_size = 69   # 输入特征的维度
    normalization = True  # 是否进行归一化

    # 其他配置
    save_model_path = './model.pth'  # 模型保存路径
    log_interval = 10  # 日志记录间隔