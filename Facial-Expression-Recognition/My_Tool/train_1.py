import utils
import torch
from d2l import torch as d2l
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
from torch.autograd import Variable
from models import *


use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

use_cuda = torch.cuda.is_available()

def train_end(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer_adam = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # loss函数
    criterion = nn.CrossEntropyLoss()
    loss = nn.CrossEntropyLoss()

    # 一个画画的东西
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # sum of training loss, sum of training accuracy no. of examples
        metric = d2l.Accumulator(3)
        net.train()

        if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
            frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
            decay_factor = learning_rate_decay_rate ** frac
            current_lr = lr * decay_factor
            utils.set_lr(optimizer, current_lr)  # set the decayed rate
        else:
            current_lr = lr

        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                d2l.astype(y_hat, y.dtype)
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (1 + i) / num_batches,
                             (train_l, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu()
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'on {str(device)}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')