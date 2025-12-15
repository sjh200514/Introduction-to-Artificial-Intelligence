#导入必要的库
import gym
import torch
import torch.nn as nn
import time
import pygame
from torch import optim
from torch.distributions import Categorical

#定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #第一层全连接层，输入4个特征，输出128个特征
        self.fc1 = nn.Linear(in_features=4, out_features=128)
        #第二层全连接层，输入128个特征，输出2个动作的概率 
        self.fc2 = nn.Linear(128, 2)
        #Dropout层，用于减少过拟合，丢弃60%的神经元
        self.drop = nn.Dropout(p=0.6)

    def forward(self, x):
        #通过第一层全连接层
        x = self.fc1(x)
        #应用Dropout  
        x = self.drop(x)  
        #应用ReLU激活函数
        x = nn.functional.relu(x)
        #通过第二层全连接层
        x = self.fc2(x)  
        #使用softmax函数将输出转换为概率分布
        return nn.functional.softmax(x, dim=1)

#基于REINFORCE算法的损失函数
def accumulate_loss(n, log_prob):
    #创建一个递减的奖励序列
    reward = torch.arange(n, 0, -1).float()
    #对奖励进行标准化处理 
    reward = (reward - reward.mean()) / reward.std()  
    loss = 0
    #基于每一步的log概率和对应的奖励计算总损失
    for p, r in zip(log_prob, reward):
        loss -= p * r
    return loss

if __name__ == '__main__':
    #训练模型

    #创建一个CartPole环境
    env = gym.make("CartPole-v1")
    #实例化神经网络模型
    net = Net()  
    #优化器
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    #训练循环
    for episod in range(1, 1001):  #训练1000个回合
        train_state, _ = env.reset()  #重置环境，获取初始状态
        train_step = 0  #记录当前回合的步数
        log_prob = []  #存储每一步的log概率
        for train_step in range(1, 10001):  #每个回合最多10000步
            train_state = torch.from_numpy(train_state).float().unsqueeze(0)
            probs = net(train_state)  #通过神经网络计算动作概率
            m = Categorical(probs)  #创建一个概率分布
            action = m.sample()  #根据概率分布采样一个动作
            train_state, _, done, _, _ = env.step(action.item())  #执行动作，获取下一步的状态
            if done:  #如果回合结束，跳出循环
              break
            log_prob.append(m.log_prob(action))  #存储当前动作的log概率
        if train_step > 5000:  # 如果某个回合超过5000步，则结束训练
            print(f"Last episode {episod} Run steps {train_step}")
            break
        #梯度下降算法
        optimizer.zero_grad()  #清空梯度
        loss = accumulate_loss(train_step, log_prob)  #计算损失
        loss.backward()  #反向传播
        optimizer.step()  #更新模型参数
        if episod % 10 == 0:  #每10个回合打印一次信息
            print(f"Episode {episod} Run step {train_step}")
    print("Finish training!")
    #测试模型
    #初始化pygame
    pygame.init()
    #创建一个支持图形渲染的环境
    env = gym.make('CartPole-v1', render_mode="human")
    state, _ = env.reset()  #重置环境，获取初始状态
    test_step = 0  #记录测试时的步数
    start = time.time()  #记录开始时间
    for test_step in range(1, 2001):  #测试最多2000步
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = net(state)  #通过神经网络计算动作概率
        action = torch.argmax(probs, dim=1).item()  #选择概率最高的动作
        state, _, done, _, _ = env.step(action)  #执行动作，获取下一步的状态
        if done:  #如果游戏结束，跳出循环
            break
        print(f"step = {test_step} action = {action} position = {state[0]:.2f} cart_speed = {state[1]:.2f} angle = {state[2]:.2f}  pole_speed = {state[3]:.2f}")  # 打印当前步骤的信息
    end = time.time()  #记录结束时间
    print(f"You play {end - start:.2f} seconds, {test_step} steps.")
    env.close()  #关闭环境