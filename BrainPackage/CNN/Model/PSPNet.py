import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
 
# from utils import initialize_weights
# from utils.misc import Conv2dDeformable

class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:      
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out

class PSPNet(nn.Module):
    def __init__(self, num_classes, pretrained=True,  is_need_sigmoid = True, use_aux=False, colordim =3):
        super(PSPNet, self).__init__()
        self.is_preprocess = False
        if colordim !=3:
            self.is_preprocess = True
            self.preprocess = nn.Transpose2d(colordim, 3, kernel_size=1, stride=1)
        self.is_need_sigmoid = is_need_sigmoid
        self.use_aux = use_aux
        resnet = models.resnet101(pretrained)     
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

#         if use_aux:
#             self.aux_logits = nn.Conv2d(1024, num_classes, kernel_size=1)
#             initialize_weights(self.aux_logits)
# # 初始化权重
#         initialize_weights(self.ppm, self.final)

    def forward(self, x):
        if self.is_preprocess:
            x = self.preprocess(x)
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # if self.training and self.use_aux:
        #     aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.ppm(x)
        score = self.final(x)
        score = F.upsample(score, x_size[2:], mode='bilinear') 
        # if self.training and self.use_aux:
        #     return F.upsample(x, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:], mode='bilinear')
        # return F.upsample(x, x_size[2:], mode='bilinear')
        if self.is_need_sigmoid == True:
            score = nn.functional.sigmoid(score)         
        return score
        

