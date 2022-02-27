import torch
import torch.nn as nn


class MnistFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 2500), nn.BatchNorm1d(2500), nn.ReLU(),
                                 nn.Linear(2500, 2000), nn.BatchNorm1d(2000), nn.ReLU(),
                                 nn.Linear(2000, 1500), nn.BatchNorm1d(1500), nn.ReLU(),
                                 nn.Linear(1500, 1000), nn.BatchNorm1d(1000), nn.ReLU(),
                                 nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(),
                                 nn.Linear(500, 10))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.net(x)
        return x


class AllCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, padding=2),  # 34
                                 nn.BatchNorm2d(96),
                                 nn.ReLU(),
                                 nn.Conv2d(96, 96, kernel_size=3, padding=1),  # 34
                                 nn.BatchNorm2d(96),
                                 nn.ReLU(),
                                 nn.Conv2d(96, 192, kernel_size=3, stride=2),  # 16
                                 nn.BatchNorm2d(192),
                                 nn.ReLU(),
                                 nn.Conv2d(192, 192, kernel_size=3, padding=1),  # 16
                                 nn.BatchNorm2d(192),
                                 nn.ReLU(),
                                 nn.Conv2d(192, 192, kernel_size=3, stride=2),  # 7
                                 nn.BatchNorm2d(192),
                                 nn.ReLU(),
                                 nn.Conv2d(192, 192, kernel_size=3),  # 5
                                 nn.BatchNorm2d(192),
                                 nn.ReLU(),
                                 nn.Conv2d(192, 192, kernel_size=1),  # 5
                                 nn.Conv2d(192, 10, kernel_size=1)  # 5
                                 )
        self.avg = nn.AvgPool2d(kernel_size=5)

    def forward(self, x):
        x = self.net(x)
        x = self.avg(x)
        x = torch.squeeze(x)
        return x



class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()

        self.expansion = 1

        layers = [2, 2, 2, 2]

        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)
