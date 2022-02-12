"""EfficientNet architecture.
See:
- https://arxiv.org/abs/1905.11946 - EfficientNet
- https://arxiv.org/abs/1801.04381 - MobileNet V2
- https://arxiv.org/abs/1905.02244 - MobileNet V3
- https://arxiv.org/abs/1709.01507 - Squeeze-and-Excitation
- https://arxiv.org/abs/1803.02579 - Concurrent spatial and channel squeeze-and-excitation
"""

import os
import wget

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import collections

from .layers import (DropConnect, SamePadConv2d, Attention, 
                    Swish, conv_bn_act, Flattener)

batch_norm_momentum=0.010000000000000009
batch_norm_epsilon=1e-3

EfficientNetParam = collections.namedtuple(
    "EfficientNetParam", ["width", "depth", "resolution", "dropout"])

EfficientNetParams = {
  "B0": EfficientNetParam(0.5, 0.8, 88, 0.2),
  "B1": EfficientNetParam(1.0, 1.1, 240, 0.2),
  "B2": EfficientNetParam(1.1, 1.2, 260, 0.3),
  "B3": EfficientNetParam(1.2, 1.4, 300, 0.3),
  "B4": EfficientNetParam(1.4, 1.8, 380, 0.4),
  "B5": EfficientNetParam(1.6, 2.2, 456, 0.4),
  "B6": EfficientNetParam(1.8, 2.6, 528, 0.5),
  "B7": EfficientNetParam(2.0, 3.1, 600, 0.5),
   }

EfficientNetUrls = {
    'B0': 'https://drive.google.com/uc?export=download&id=1rNU-wCPT_ebdc7qG1NzwSoqXLJxIL9Qz', 
    'B1': 'https://drive.google.com/uc?export=download&id=1rNU-wCPT_ebdc7qG1NzwSoqXLJxIL9Qz',
    'B2': 'https://drive.google.com/uc?export=download&id=1SeA-VuRiuWI9f8PVuhTYYdu8NjKFl_dL',
    'B3': 'https://drive.google.com/uc?export=download&id=1fhi6xlrJl1iKl2b2knxPchjCsI9f9-Fr',
    'B4': 'https://drive.google.com/uc?export=download&id=1iroAuwlUssk3mzcbHdYDGhnXLm37ZfN2',
    'B5': 'https://drive.google.com/uc?export=download&id=188XFvL4JqH8SX0Pb3EtTENdaGzolN4-s',
    'B6': 'https://drive.google.com/uc?export=download&id=1PGLFWp3xF8LVUjVGJ_h9DYYhyOAGX2gG',
    'B7': 'https://drive.google.com/uc?export=download&id=18BHuBD2HpjTj2r9ffHo_Y6dgdprmcGUc',
    }
    

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, expand_ratio, kernel_size, stride, se_reduction, drop_connect_ratio=0.2):
        """Basic building block - Inverted Residual Convolution from MobileNet V2 
        architecture.
        Arguments:
            expand_ratio (int): ratio to expand convolution in width inside convolution.
                It's not the same as width_mult in MobileNet which is used to increase
                persistent input and output number of channels in layer. Which is not a
                projection of channels inside the conv. 
        """
        super().__init__()
        
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        if self.use_res_connect:
            self.dropconnect = DropConnect(drop_connect_ratio)
            
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depth-wise 
                SamePadConv2d(inp=hidden_dim, oup=hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim, 
                              bias=False),
                nn.BatchNorm2d(hidden_dim, eps=batch_norm_epsilon, momentum=batch_norm_momentum),
                Swish(), 
                Attention(channels=hidden_dim, reduction=4),  # somehow here reduction should be always 4
                
                # point-wise-linear
                SamePadConv2d(inp=hidden_dim, oup=oup, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(oup, eps=batch_norm_epsilon, momentum=batch_norm_momentum),
            )
        else:
            self.conv = nn.Sequential(
                # point-wise
                SamePadConv2d(inp, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=batch_norm_epsilon, momentum=batch_norm_momentum),
                Swish(), 
                                
                # depth-wise
                SamePadConv2d(hidden_dim, hidden_dim, kernel_size, stride, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=batch_norm_epsilon, momentum=batch_norm_momentum),
                Swish(),
                Attention(channels=hidden_dim, reduction=se_reduction),  
                
                # point-wise-linear
                SamePadConv2d(hidden_dim, oup, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(oup, eps=batch_norm_epsilon, momentum=batch_norm_momentum),
            )

    def forward(self, inputs):
        if self.use_res_connect: 
            return self.dropconnect(inputs) + self.conv(inputs)
        else: 
            return self.conv(inputs)

        
def round_filters(filters, width_coef, depth_divisor=8, min_depth=None):
    """Calculate and round number of filters based on depth multiplier. """
    if not width_coef:
        return filters
    filters *= width_coef
    min_depth = min_depth or depth_divisor
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coef):
    """Round number of filters based on depth multiplier."""
    if not depth_coef:
        return repeats
    return int(math.ceil(depth_coef * repeats)) 
    
        
class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_coef=1., depth_coef=1., scale=1., dropout_ratio=0.2, 
                 se_reduction=24, drop_connect_ratio=0.5):
        super(EfficientNet, self).__init__()
        
        self.num_classes = num_classes
        block = InvertedResidual
        input_channel     = round_filters(32, width_coef)   # input_channel = round(32*width_coef) 
        self.last_channel = round_filters(1280, width_coef)     # self.last_channel  = round(1280*width_coef)
        config = np.array([
            # stride only for first layer in group, all other always with stride 1
            # channel,expand,repeat,stride,kernel_size
            [16,  1, 1, 1, 3],
            [24,  6, 2, 2, 3],
            [40,  6, 2, 2, 5],
            [80,  6, 3, 2, 3],
            [112, 6, 3, 1, 5],
            [192, 6, 4, 2, 5],
            [320, 6, 1, 1, 3],
            ])

        # first steam layer - ordinar conv
        self.features = [conv_bn_act(3, input_channel, kernel_size=3, stride=2, bias=False)]

        # main 7 group of layers
        for c, t, n, s, k in config:
            output_channel = round_filters(c, width_coef)
            for i in range(round_repeats(n, depth_coef)):
                if i == 0:
                    self.features.append(block(inp=input_channel, oup=output_channel, expand_ratio=t, kernel_size=k, 
                                               stride=s, se_reduction=se_reduction, drop_connect_ratio=drop_connect_ratio))
                else:
                    # here stride is equal 1 because only first layer in group could have stride 2, 
                    self.features.append(block(inp=input_channel, oup=output_channel, expand_ratio=t, kernel_size=k, 
                                               stride=1, se_reduction=se_reduction, drop_connect_ratio=drop_connect_ratio))
                input_channel = output_channel
        
        # building last several layers
        self.features.append(conv_bn_act(input_channel, self.last_channel, kernel_size=1, bias=False))
        self.features = nn.Sequential(*self.features)
            
        self.fc_adv = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
            )

        self.fc_val = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, 1),
            )
     
        self._initialize_weights()

                      
    def _initialize_weights(self):
        flattener = Flattener()
        flattened = flattener(self)
        for m in flattened:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(x.size(0), self.num_classes)

        output = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_classes)
        return output
    
    
def efficientnet(num_classes, net="B0", pretrained=False):
    """Weights for B5-B7 models should be loaded manually since after 100Mb 
    Google Drive checks virusis. I'm lazzy to fix it sice I never user models
    higher that B4 :)
    """
    net = net.upper()
    model = EfficientNet(num_classes=num_classes,
        width_coef    = EfficientNetParams[net].width, 
        depth_coef    = EfficientNetParams[net].depth, 
        scale         = EfficientNetParams[net].resolution, 
        se_reduction  = 24, 
        dropout_ratio = EfficientNetParams[net].dropout, 
    )
    
    if pretrained:
        assert net not in ['B5', 'B6', 'B7'], f'Weights for this model should be loaded manualy from {EfficientNetUrls[net]}'  
        # create folder for weights if it doesn't exist
        if not os.path.exists('weights'): 
            os.makedirs('weights')
        
        # download weights if needed, progress bar in jupyter for wget currently doesn't implemented
        weights_path = f'weights/efficientnet-{net}.pth'.lower()
        if not os.path.isfile(weights_path):
            wget.download(url=EfficientNetUrls[net], out=weights_path)
                    
        checkpoint = torch.load(weights_path)
        filtered_state_dict = {k:v for k, v in checkpoint.items() if 'features' in k}
        model.load_state_dict(filtered_state_dict, strict=False)
        
    return model