class Config:
    # 模型超参数
    hidden_size = 64  # 隐藏层大小
    output_size = 1   # 输出层大小（对于二分类任务）
    learning_rate = 0.01  # 学习率
    num_epochs = 1000  # 训练轮数
    batch_size = 32   # 批次大小

    # 数据相关参数
    input_size = 69   # 输入特征的维度
    normalization = True  # 是否进行归一化

    # 其他配置
    save_model_path = './model.pth'  # 模型保存路径
    log_interval = 10  # 日志记录间隔