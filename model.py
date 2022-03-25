'''
The model of DNN to fit function g in Pytorch
The Loss function is defined by the negative reward function C in paper
The training data is sampled from the Monta Carlo method  as the paper shows
Config is the class covering the parameters and hyperparameters
'''

import torch
import numpy as np
import time
from torch import nn
import os

class myModel(nn.Module): # build the DNN model

    # The num of hidden layers is chosen as 3, activation function Relu
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config.in_dim, config.hidden_1),
            nn.ReLU(True),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.ReLU(True),
            nn.Linear(config.hidden_2, config.hidden_3),
            nn.ReLU(True),
            nn.Linear(config.hidden_3, config.out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

def myLoss(config, f, g_theta1, g_theta2, g_S): # definition of the Loss function

    r = torch.tensor(config.r)
    t = torch.tensor(config.t)
    T = torch.tensor(config.T)

    # Compute the coefficients
    beta1 = torch.exp(-r * (T - t))
    beta2 = T -  t
    f = f
    g_theta1 = g_theta1
    g_theta2 = g_theta2
    g_S = g_S

    C1 = beta1 * torch.mul(f, g_theta1)
    C2 = beta2 * torch.mul(g_S, g_theta2)
    loss = torch.mean(C1 - C2)
    return loss


class Config:    # Configuration of the hyperparameters and parameters

    in_dim = 2
    out_dim = 1
    hidden_1 = 3
    hidden_2 = 4
    hidden_3 =3
    epoch = 100
    save_path = "./checkpoint"  #the path that the weights are saved

    def __init__(self, t, x, T, K, M, r, delta,):
        self.x = x
        self.t = t
        self.T = T
        self.K = K
        self.M = M
        self.r = r
        self.delta = delta

    def f(self, X):   # compute

        K = torch.tensor(self.K)
        return X - K

    def g_s(self, U, X):
        r = torch.tensor(self.r)
        t = torch.tensor(self.t)
        K = torch.tensor(self.K)
        temp1 = r * (X + 1 - K)
        temp2 = torch.exp(-r * (U - t))
        return torch.mul(temp1, temp2)



def loadData(config):
    t = config.t
    T = config.T
    delta = config.delta
    x = config.x
    r = config.r


    U = np.array([])
    X1 = np.array([])
    X2 = np.array([])
    for i in range(config.M):
        u = np.random.uniform(t, T)
        Z1 = np.random.randn()
        Z2 = np.random.randn()
        x1 = x * np.exp((r - np.power(delta, 2) / 2) * (u - t) +
                        delta * Z1 * np.sqrt(u - t))
        x2 = x1 * np.exp((r - np.power(delta, 2) / 2) * (T - u) +
                         delta * Z2 * np.sqrt(T - u))
        U = np.append(U,u)
        X1 = np.append(X1,x1)
        X2 = np.append(X2,x2)
    U = torch.from_numpy(U)
    X1 = torch.from_numpy(X1)
    X2 = torch.from_numpy(X2)
    return U, X1, X2


def get_train_data(U, X1, X2):
    U = torch.unsqueeze(U, dim=0)
    X1 = torch.unsqueeze(X1,dim=0)
    X2 = torch.unsqueeze(X2, dim=0)
    train_U = torch.cat([U, X1], dim=0).t()
    train_T = torch.cat([U, X2], dim=0).t()
    return train_U, train_T



def train(config, train_data):

    U, X_U, X_T = train_data
    train_U, train_T = get_train_data(U, X_U, X_T)
    train_U = train_U.to(torch.float32)
    train_T = train_T.to(torch.float32)

    net = myModel(config)
    f = config.f(X_T)
    g_s = config.g_s(U, X_U)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()

    for i in range(config.epoch):

        optimizer.zero_grad()
        g_theta1 = net(train_T)
        g_theta2 = net(train_U)
        loss = -myLoss(config, f, g_theta1, g_theta2, g_s)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(loss.item())
    if not os.path.exists(config.save_path):
            os.mkdirs(config.save_path)
    torch.save(net.state_dict(), config.save_path + '/weights.pth')



def predict(config, predict_data):
    U, X_U, X_T = predict_data
    predict_U, predict_T = get_train_data(U, X_U, X_T)
    predict_U = predict_U.to(torch.float32)
    predict_T = predict_T.to(torch.float32)

    net = myModel(config)
    net.load_state_dict(torch.load(config.save_path + '/weights.pth'))
    f = config.f(X_T)
    g_s = config.g_s(U, X_U)
    net.eval()
    g_theta1 = net(predict_T)
    g_theta2 = net(predict_U)
    reward = myLoss(config, f, g_theta1, g_theta2, g_s)


    return reward