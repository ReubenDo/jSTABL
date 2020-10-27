# Adapted from Orbes-Arteaga, et al 2019. Multi-domain adaptation in brain mri through paired con-sistency and adversarial learning. https://arxiv.org/abs/1908.05959
import torch 
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, nb_channels, nb_classes, base_num_features, shape):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(nb_channels, base_num_features, kernel_size=3, stride=2)
        self.IN1 = nn.InstanceNorm2d(base_num_features)
        self.conv2 = nn.Conv2d(base_num_features, 2*base_num_features, kernel_size=3, stride=2)
        self.IN2 = nn.InstanceNorm2d(2*base_num_features)
        self.conv3 = nn.Conv2d(2*base_num_features, 4*base_num_features, kernel_size=3, stride=2)
        self.IN3 = nn.InstanceNorm2d(4*base_num_features)
        self.conv4 = nn.Conv2d(4*base_num_features, 8*base_num_features, kernel_size=3, stride=2)
        self.IN4 = nn.InstanceNorm2d(8*base_num_features)
        self.fc1 = nn.Linear(8*base_num_features*shape[0]*shape[1], 16*base_num_features)
        self.drop_1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16*base_num_features, 8*base_num_features)
        self.drop_2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(8*base_num_features, nb_classes)

        self.base_num_features = base_num_features
        self.shape = shape

    def forward(self, x):
        x = F.leaky_relu(self.IN1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.IN2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.IN3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.IN4(self.conv4(x)), 0.2)

        x = x.view(-1, 8*self.base_num_features*self.shape[0]*self.shape[1])
        x = F.relu(self.drop_1(self.fc1(x)))
        x = F.relu(self.drop_2(self.fc2(x)))
        x = self.fc3(x)

        return x