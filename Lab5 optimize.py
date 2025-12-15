#对算法性能优化过后的版本
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#加载和预处理 CIFAR-10 数据集
#将图像转换为张量并标准化
transform = transforms.Compose(
    [transforms.ToTensor(),  # 转换为张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 标准化

#加载训练数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

#加载测试数据集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)

#CIFAR-10 数据集的类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#模型保存路径
PATH = './cifar_net.pth'

#设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#显示图像的函数
def imshow(img):
    img = img / 2 + 0.5     #反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  #转换通道顺序以匹配matplotlib
    plt.show()

#定义卷积神经网络（CNN）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        #新添加的卷积层
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        #计算全连接层的输入尺寸
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)  #展平为一维向量，匹配fc1的输入尺寸
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#主程序
if __name__ == '__main__':
    #初始化网络并移动到设备上
    net = Net().to(device)
    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    #训练循环
    for epoch in range(10):  #训练10次
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            #获取输入和标签，并移动到设备上
            inputs, labels = data[0].to(device), data[1].to(device)

            #清零参数梯度
            optimizer.zero_grad()

            #前向传播、反向传播和优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #打印统计信息
            running_loss += loss.item()
            if i % 2000 == 1999:  #每2000个mini-batch打印一次
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    #保存训练好的模型参数
    torch.save(net.state_dict(), PATH)
    print('Finished Training')
    #显示一些随机训练图像和预测结果
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    #显示图像
    imshow(torchvision.utils.make_grid(images))
    #打印类别
    print('GroundTruth: ',' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    images = images.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
    #计算测试集上的准确率
    correct = 0
    total = 0
    with torch.no_grad():  #测试时不计算梯度
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            #通过网络计算输出
            outputs = net(images)
            #获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    
    #准备计算每个类别的预测准确率
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    #测试时不计算梯度
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # 收集每个类别的正确预测数量
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    #打印每个类别的准确率
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')