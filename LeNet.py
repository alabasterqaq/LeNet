import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os


class MyLeNet5(nn.Module):
    # 初始化网络
    def __init__(self):
        super(MyLeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5), nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, 10)
        )

    # 前向传播
    def forward(self, x):
        y = self.net(x)
        return y

# 将图像转换为张量形式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(
    root='D:\Programming_software\Dataset',  # 下载路径
    train=True,  # 是训练集
    download=True,  # 如果该路径没有该数据集，则进行下载
    transform=data_transform  # 数据集转换参数
)

# 批次加载器
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(
    root='D:\Programming_software\Dataset',  # 下载路径
    train=False,  # 是训练集
    download=True,  # 如果该路径没有该数据集，则进行下载
    transform=data_transform  # 数据集转换参数
)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 判断是否有gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用net，将模型数据转移到gpu
model = MyLeNet5().to(device)

# 选择损失函数
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数，自带Softmax激活函数

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率每隔10轮次， 变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 定于训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, dim=1)
        # 计算当前轮次时，训练集的精确度
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    print("train_loss: ", str(loss / n))
    print("train_acc: ", str(current / n))


def test(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    # 该局部关闭梯度计算功能，提高运算效率
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # 前向传播
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, dim=1)
            # 计算当前轮次时，训练集的精确度
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print("test_loss: ", str(loss / n))
        print("test_acc: ", str(current / n))
        return current / n  # 返回精确度


# 开始训练
epoch = 20
max_acc = 0
for t in range(epoch):
    print(f"epoch{t + 1}\n---------------")
    train(train_dataloader, model, loss_fn, optimizer)
    a = test(test_dataloader, model, loss_fn)
    # 保存最好的模型参数
    if a > max_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir(folder)
        max_acc = a
        print("current best model acc = ", a)
        torch.save(model.state_dict(), 'save_model/best_model.pth')
print("Done!")
