import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.GroupNorm(32, planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32, planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

        gn_init(self.bn1)
        gn_init(self.bn2)
        gn_init(self.bn3, zero_init=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv2d_init(m):
    assert isinstance(m, nn.Conv2d)
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))

def gn_init(m, zero_init=False):
    assert isinstance(m, nn.GroupNorm)
    m.weight.data.fill_(0. if zero_init else 1.)
    m.bias.data.zero_()



class ResNetMOD3AUTO(nn.Module):
    def __init__(self, block, layers, output_stride, pretrained=True):
        self.inplanes = 64
        super(ResNetMOD3AUTO, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]

        self.conv0 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                                             bias=True),nn.PReLU(32))

        self.conv1a = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1,
                                              bias=True),nn.PReLU(32),nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),nn.PReLU(64))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                                             bias=True))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion),
            )
            m = downsample[1]
            assert isinstance(m, nn.GroupNorm)
            gn_init(m)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation = dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation=dilation))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32,planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation))

        return nn.Sequential(*layers)


    def forward(self, x,return_feat=False,return_feat_aux=False):
        conv_out = [x]
        x0=self.conv0(x)
        conv_out.append(x0)
        x = self.conv1a(x0)

        conv_out.append(x)
        x_ = self.conv2(x)
        x = self.layer1(x_)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        if return_feat_aux:
            return conv_out,x_

        x = self.layer4(x)
        conv_out.append(x)
        if return_feat:
            return conv_out,x_
        return x

    def _load_pretrained_model(self):
        import torch
        pretrain_dict = torch.load('res50gn.pth',map_location='cpu')['state_dict']
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            k_=k[7:]
            if k_ in state_dict:
                print(k_)
                model_dict[k_] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict,strict=False)


def ResNet50_MOD3AT(os,pretrained=False):
    model = ResNetMOD3AUTO(Bottleneck, [3, 4, 6, 3], os,  pretrained=pretrained)
    return model
