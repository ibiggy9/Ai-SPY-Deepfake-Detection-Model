import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class CNNTest(nn.Module):
    def __init__(self):
        super(CNNTest, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(8,4), padding=(4,2))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(4)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(6,3), padding=(3,2))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(4,2), padding=(2,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(4,2), padding=(2,1))
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4,2), padding=(2,1))
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(2,2), padding=(2,2)) 
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(2,2), padding=(1,1)) 
        self.bn7 = nn.BatchNorm2d(1024)
        self.pool7 = nn.MaxPool2d(2)

        self.drop = nn.Dropout(0.1)
        self.fc1 = nn.Linear(2816, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.leaky_relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.leaky_relu(self.bn5(self.conv5(x))))
        #x = self.pool6(F.leaky_relu(self.bn6(self.conv6(x))))
        #x = self.pool7(F.leaky_relu(self.bn7(self.conv7(x))))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return x