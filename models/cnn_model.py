import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class CNNTest(nn.Module):
    def __init__(self):
        super(CNNTest, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(40)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(40, 60, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(60)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(60, 80, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(80)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(80, 100, kernel_size=2, padding=1)
        self.bn4 = nn.BatchNorm2d(100)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(100, 200, kernel_size=2, padding=1)
        self.bn5 = nn.BatchNorm2d(200)
        self.pool5 = nn.MaxPool2d(2)

   
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(19200, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.leaky_relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.leaky_relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return x