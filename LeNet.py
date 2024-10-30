import torch
from torch import nn
 
# 定义网络模型
class MyLeNet5(nn.Module):
    # 初始化网络
    def __init__(self):
        super(MyLeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),  nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120, 84),  nn.Tanh(),
            nn.Linear(84, 10)
        )
 
    # 前向传播
    def forward(self, x):
        y = self.net(x)
        return y
 
#测试输出
if __name__ == '__main__': 
    x1 = torch.rand([1, 1, 28, 28])
    model = MyLeNet5()
    y1 = model(x1)
    print(x1)
    print(y1)
