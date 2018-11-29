
import numpy as np
import torch
import torch.nn as nn
import random

class CNN(nn.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.relu = nn.LeakyReLU()
        self.dropout=nn.Dropout(0.25)
        
        #input channels = 6, output = 48, sizeOut 160x120
        self.conv1 = nn.Conv3d(6, 48, kernel_size=4, stride=1, padding=1)
        self.bnorm1=nn.BatchNorm3d(48)
        
        #input channels = 48, output = 96, sizeOut 156x116
        self.conv2 = nn.Conv3d(48, 96, kernel_size=6, stride=1, padding=0)
        self.bnorm2 = nn.BatchNorm3d(96)

        #input channels = 96, output = 96, sizeOut 78x58
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        #input cahnnels = 96, output =96, sizeOut 74x54
        self.conv3 = nn.Conv3d(96, 96, kernel_size=6, stride=1, padding=0)
        self.bnorm3 = nn.BatchNorm3d(96)
        
        #input channels = 96, output = 96, sizeOut 37x27
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        #input channels = 96, output = 96, sizeout 33x23
        self.conv4 = nn.Conv3d(96, 96, kernel_size=5, stride=1, padding=0)
        self.bnorm4 = nn.BatchNorm3d(96)
        
        #input channels = 96, output = 96, sizeOut 16x11
        self.pool3=nn.MaxPool3d(kernel_size=2, stride =2, padding=0)
        
        #first fully connected
        self.fc1 = nn.Linear(16896, 1024)
        
        #second fully connected
        self.fc2 = nn.Linear(1024, 32)
        
        #third fully connected
        self.fc3=nn.Linear(32,4)
        
    def forward(self, x):

        x = self.bnorm1(self.relu(self.conv1(x)))
        x = self.bnorm2(self.relu(self.conv2(x)))
        x = self.relu(self.pool1(x))
        x = self.bnorm3(self.relu(self.conv3(x)))
        x = self.relu(self.pool2(x))
        x = self.bnorm4(self.relu(self.conv4(x)))
        x = self.relu(self.pool3(x))

        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # flatten

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


Transition = namedtuple('Transition',
                        ('state','action','next_state','reward'))    

class ReplayMaemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)