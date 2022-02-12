'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-11-16 10:37:59
LastEditors: Zhang
LastEditTime: 2021-12-08 22:50:57
'''

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, padding=1, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=out_channels, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        # print(self.residual_function(x).shape)
        # print(self.shortcut(x).shape)
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=out_channels, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels*BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        nn.ReLU(inplace=True)(self.residual_function(x)+self.shortcut(x))


class DQN(nn.Module):
    def __init__(self, in_channels, block, num_blocks, num_actions):
        super(DQN, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,groups=in_channels, padding=1, bias=False),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc = nn.Linear(512 * block.expansion, num_actions)

    def _make_layer(self, block, out_channels, num_block, stride):

        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x):

        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, block, num_blocks, num_actions):
        super(Dueling_DQN,self).__init__()
        self.in_channels = 64
        self.num_actions = num_actions

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,groups=in_channels, padding=1, bias=False),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc_adv = nn.Linear(512 * block.expansion, num_actions)
        self.fc_val = nn.Linear(512 * block.expansion, 1)

    def _make_layer(self, block, out_channels, num_block, stride):

        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        adv = self.fc_adv(output)
        val = self.fc_val(output).expand(output.size(0), self.num_actions)

        output = val + adv - adv.mean(1).unsqueeze(1).expand(output.size(0), self.num_actions)

        return output


def dqn_res18(in_channels, num_actions):
    return DQN(in_channels=in_channels, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_actions=num_actions)

def ddqn_res18(in_channels, num_actions):
    return Dueling_DQN(in_channels=in_channels, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_actions=num_actions)