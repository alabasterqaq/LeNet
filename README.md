# LeNet

数据集通过包torchvision中的datasets库进行下载。

选用的损失函数是交叉熵损失函数。

每训练一个循环epoch，就进行一次测试，实时显示一定轮次后训练集和测试集的拟合情况。设置的epoch为20
