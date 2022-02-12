'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-12-08 23:10:35
LastEditors: Zhang
LastEditTime: 2021-12-08 23:10:52
'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# def swish(x):
#     return x * torch.sigmoid(x)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7, activation=torch.nn.ReLU6):
        super(CBAMLayer, self).__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # channel attention
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, activation=torch.nn.ReLU6):
        """Channel attention."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, stride=1, padding=0, bias=True),
            activation(),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()
      
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        return x
    
    
class Attention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(channels//reduction, channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.attention(x))
        
    
class Flatten(nn.Module):      
    def forward(self, inp): return inp.view(inp.size(0), -1)

    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
    
class DropConnect(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        return x
#         if not self.training:
#             return x

#         random_tensor = self.ratio
#         random_tensor += torch.rand([x.shape[0], 1, 1, 1], dtype=torch.float, device=x.device)
#         random_tensor.requires_grad_(False)
#         return x / self.ratio * random_tensor.floor()
    

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, output_size):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(output_size)
        self.mp = nn.AdaptiveMaxPool2d(output_size)
        
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)
        
            
def custom_head(num_classes=1000, num_feat=1024, ps=0.5):
    """Head leveraged from fast.ai library. Dropout assigned in params.json.
    ps assigned to last fc layer. The layer before has ps/2. The same as in 
    fast.ai."""
    return nn.Sequential(
        Flatten(),
        nn.BatchNorm1d(num_features=num_feat),
        nn.Dropout(p=ps/2),
        nn.Linear(in_features=num_feat, out_features=num_feat // 2, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(num_features=num_feat // 2),
        nn.Dropout(p=ps),
        nn.Linear(in_features=num_feat // 2, out_features=num_classes, bias=True),
    )


class Upsample(nn.Module):
    def __init__(self, scale):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
    

def conv_bn_act(inp, oup, kernel_size, stride=1, groups=1, bias=True, eps=1e-3, momentum=0.01):
    return nn.Sequential(
        SamePadConv2d(inp, oup, kernel_size, stride, groups=groups, bias=bias),
        nn.BatchNorm2d(oup, eps, momentum),
        Swish()
    )


class SamePadConv2d(nn.Conv2d):
    """
    Conv with TF padding='same'
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
    It means if we have kernel_size=5 and we want to leave conv output
    be the same size as input we need to define how many pads should we have.
    In TF we have option padding='same' but in Pytorch we need to provide
    number of paddings. So we class is analog of TF padding='same'.
    As you can see we don't have `padding` as input to this class, since 
    it we be calc automaticaly.
    """

    def __init__(self, inp, oup, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__(inp, oup, kernel_size, stride, 0, dilation, groups, bias, padding_mode)

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_)
        rows_odd = (padding_rows % 2 != 0)
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0])
        padding_cols, cols_odd = self.get_pad_odd(x.shape[3], self.weight.shape[3], self.stride[1], self.dilation[1])

        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(x, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)


class Flattener:
    """Flatten modules in a list. Even with modules in modules.
    Works recursively.
    
    Example:
    
    flatener = Flattener()
    flatener(model.features[15])
    
    Returns a list of all layers in a module.
    """
    def __init__(self):
        self.flattened_module = []
    
    def flat(self, module):
        flattened_module = []
        childrens = list(module.children())
        for children in childrens:
            flattened_module.append(children)
        return flattened_module
    
    def __call__(self, module):
        """Recursive function."""
        childrens = list(module.children())
        for children in childrens:
            if len(self.flat(children))==0:
                self.flattened_module.append(children)
            else:
                self.__call__(children) 
        return self.flattened_module