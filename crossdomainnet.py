from __future__ import division
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
 
 
class CrossDomainNet(nn.Module):
 
    def __init__(self):

        # Layer 1 - conv(3, 9, 9, 24)
        # Layer 2 - conv(24, 5, 5, 48)
        # Layer 3 - conv(48, 3, 3, 64)
        # Source and Target - (80,160)
 
        super(CrossDomainNet, self).__init__()
 
        self.conv1 = nn.Conv2d(3, 24, 5, padding=2)
        self.conv2 = nn.Conv2d(24, 48, 3, padding=1)
        self.conv3 = nn.Conv2d(48,64,3, padding=1)
        self.bn24 = nn.BatchNorm2d(24, eps=1e-3)
        self.bn48 = nn.BatchNorm2d(48, eps=1e-3)
        self.bn64 = nn.BatchNorm2d(64, eps=1e-3)

        self.dconv1 = nn.ConvTranspose2d(64, 48, 3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(48, 24, 3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(24, 3, 5, padding=2)
        self.dbn24 = nn.BatchNorm2d(24, eps=1e-3)
        self.dbn48 = nn.BatchNorm2d(48, eps=1e-3)
        self.dbn64 = nn.BatchNorm2d(64, eps=1e-3)

        self.pool22 = nn.MaxPool2d(2, 2)
        self.pool21 = nn.MaxPool2d(2, 1)
        self.unpool22 = nn.MaxUnpool2d(2, 2)
        self.unpool21 = nn.MaxUnpool2d(2, 1)

 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # print m.kernel_size[0], m.kernel_size[1], m.out_channels
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # print m.kernel_size[0], m.kernel_size[1], m.out_channels
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
    def forward(self, x):
        x = self.pool22(F.relu(self.bn24(self.conv1(x))))
        x = self.pool22(F.relu(self.bn48(self.conv2(x))))
        x = self.pool21(F.relu(self.bn64(self.conv3(x))))
        x = self.unpool21(x)
        x = F.relu(x)
        x = self.dbn64(x)
        x = self.dconv1(x)
        x = self.dconv2(self.dbn48(F.relu(self.unpool22(x))))
        x = self.dconv3(self.dbn24(F.relu(self.unpool22(x))))
        return x

class Discriminator(nn.Module):
 
    def __init__(self):

        # Layer 1 - conv(3, 9, 9, 24)
        # Layer 2 - conv(24, 5, 5, 48)
        # Layer 3 - conv(48, 3, 3, 64)
        # Source and Target - (80,160)
 
        super(Discriminator, self).__init__()
 
        self.conv1 = nn.Conv2d(3, 24, 5, padding=2)
        self.conv2 = nn.Conv2d(24, 48, 3, padding=1)
        self.conv3 = nn.Conv2d(48,64,3, padding=1)
        self.bn24 = nn.BatchNorm2d(24, eps=1e-3)
        self.bn48 = nn.BatchNorm2d(48, eps=1e-3)
        self.bn64 = nn.BatchNorm2d(64, eps=1e-3)

        self.pool22 = nn.MaxPool2d(2, 2)
        self.pool21 = nn.MaxPool2d(2, 1)

        self.fc1 = nn.Linear(64*19*39, 100)
        self.bnfc1 = nn.BatchNorm1d(100, eps=1e-3)
 
        self.fc2 = nn.Linear(100, 2)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
    def forward(self, x):
        x = F.relu(self.bn24(self.conv1(x)))
        x = self.pool22(x)
        x = self.pool22(F.relu(self.bn48(self.conv2(x))))
        x = self.pool21(F.relu(self.bn64(self.conv3(x))))
        x = x.view(-1, 64*39*19)
        x = F.relu(self.bnfc1(self.fc1(x)))
        x = self.fc2(x)
        return x

Nets=CrossDomainNet()

X=Variable(torch.rand(20,3,80,160))

Y=Nets.forward(X)
print Y.size()



