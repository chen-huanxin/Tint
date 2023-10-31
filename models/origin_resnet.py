import torch.nn as nn
import torch
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

class MSNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):

        super(MSNet, self).__init__()
        self.include_top = include_top


        self.groups = groups
        self.width_per_group = width_per_group

        self.in_channel = 64
        self.input_c = 3
        self.conv1 = nn.Conv2d(self.input_c, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #self.spatialChannel = 2048  # ResNet50
        self.spatialChannel = 512   # ResNet32

        self.layer1_1 = self._make_layer(block, 64, blocks_num[0])
        self.layer1_2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer1_3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer1_4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.conv1_a = nn.Conv2d(int(self.spatialChannel), int(self.spatialChannel/2), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_a = nn.BatchNorm2d(int(self.spatialChannel/2))
        self.conv1_b = nn.Conv2d(int(self.spatialChannel/2), 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_b = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

        self.in_channel = 64
        self.conv2 = nn.Conv2d(self.input_c, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(self.in_channel)

        self.layer2_1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2_2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer2_3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer2_4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.conv2_a = nn.Conv2d(int(self.spatialChannel), int(self.spatialChannel/2), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_a = nn.BatchNorm2d(int(self.spatialChannel/2))
        self.conv2_b = nn.Conv2d(int(self.spatialChannel/2), 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_b = nn.BatchNorm2d(1)

        self.in_channel = 64
        self.conv3 = nn.Conv2d(self.input_c, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn3 = nn.BatchNorm2d(self.in_channel)
        self.layer3_1 = self._make_layer(block, 64, blocks_num[0])
        self.layer3_2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3_3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer3_4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.conv3_a = nn.Conv2d(int(self.spatialChannel), int(self.spatialChannel/2), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_a = nn.BatchNorm2d(int(self.spatialChannel/2))
        self.conv3_b = nn.Conv2d(int(self.spatialChannel/2), 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_b = nn.BatchNorm2d(1)

        self.attentionFC1 = nn.Linear(int(self.spatialChannel * 3), int(self.spatialChannel * 3))
        self.attentionFC2 = nn.Linear(int(self.spatialChannel * 3), int(self.spatialChannel * 3))
        self.fc_class = nn.Linear(int(self.spatialChannel * 3), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Levels : x, 64 ,128
        # Three Level Image
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2], x.size()[3]))

        # Centre Crop local patch 64 x 64
        x_local2 = x[:, :, 80:(80 + 64), 80:(80 + 64)]
        x_local2 = nn.functional.upsample_bilinear(x_local2, (x.size()[2], x.size()[3]))

        Global = x
        Local1 = x_local
        Local2 = x_local2

        out1 = self.relu(self.bn1(self.conv1(Global)))
        out1 = self.maxpool(out1)
        out1 = self.layer1_1(out1)
        out1 = self.layer1_2(out1)
        out1 = self.layer1_3(out1)
        out1 = self.layer1_4(out1)

        out2 = self.relu(self.bn2(self.conv2(Local1)))
        out2 = self.maxpool(out2)
        out2 = self.layer2_1(out2)
        out2 = self.layer2_2(out2)
        out2 = self.layer2_3(out2)
        out2 = self.layer2_4(out2)

        out3 = self.relu(self.bn3(self.conv3(Local2)))
        out3 = self.maxpool(out3)
        out3 = self.layer3_1(out3) # torch.Size([1, 256, 56, 56])
        out3 = self.layer3_2(out3) # torch.Size([1, 512, 28, 28])
        out3 = self.layer3_3(out3) # torch.Size([1, 1024, 14, 14])
        out3 = self.layer3_4(out3) # torch.Size([1, 2048, 7, 7])

        # Channal Attention LAYER
        Channal_out1 = F.avg_pool2d(out1, out1.size()[3])
        Channal_out2 = F.avg_pool2d(out2, out2.size()[3])
        Channal_out3 = F.avg_pool2d(out3, out3.size()[3])

        # Channal Block
        OneChannalSize = Channal_out1.size()[1]
        AllChannalAttention = torch.cat([Channal_out1, Channal_out2, Channal_out3], dim=1)
        AllChannalAttention = AllChannalAttention[:, :, 0, 0]

        AllChannalAttention_1 = self.attentionFC1(AllChannalAttention)
        AllChannalAttention_1 = self.relu(AllChannalAttention_1)
        AllChannalAttention_2 = self.attentionFC2(AllChannalAttention_1)
        AllChannalAttention_2 = F.softmax(AllChannalAttention_2, dim=1)

        SplitCAttention1 = AllChannalAttention_2[:, :OneChannalSize]
        SplitCAttention2 = AllChannalAttention_2[:, OneChannalSize:OneChannalSize*2]
        SplitCAttention3 = AllChannalAttention_2[:, OneChannalSize*2:]

        SplitCAttention1 = torch.unsqueeze(SplitCAttention1, dim=-1)
        SplitCAttention1 = torch.unsqueeze(SplitCAttention1, dim=-1)

        SplitCAttention2 = torch.unsqueeze(SplitCAttention2, dim=-1)
        SplitCAttention2 = torch.unsqueeze(SplitCAttention2, dim=-1)

        SplitCAttention3 = torch.unsqueeze(SplitCAttention3, dim=-1)
        SplitCAttention3 = torch.unsqueeze(SplitCAttention3, dim=-1)

        out1 = out1 + out1 * SplitCAttention1
        out2 = out2 + out2 * SplitCAttention2
        out3 = out3 + out3 * SplitCAttention3

        # Spatial ATTENTION LAYER
        Attention1 = self.conv1_a(out1)
        Attention1 = self.bn1_a(Attention1)
        Attention1 = self.relu(Attention1)
        Attention1 = self.conv1_b(Attention1)
        Attention1 = self.bn1_b(Attention1)

        Attention2 = self.conv2_a(out2)
        Attention2 = self.bn2_a(Attention2)
        Attention2 = self.relu(Attention2)
        Attention2 = self.conv2_b(Attention2)
        Attention2 = self.bn2_b(Attention2)

        Attention3 = self.conv3_a(out3)
        Attention3 = self.bn3_a(Attention3)
        Attention3 = self.relu(Attention3)
        Attention3 = self.conv3_b(Attention3)
        Attention3 = self.bn3_b(Attention3)

        AttentionConcat = torch.concat([Attention1, Attention2, Attention3], dim=1)

        AttentionConcat = F.softmax(AttentionConcat, dim=1)

        SplitAttention1 = AttentionConcat[:, 0, :, :]
        SplitAttention2 = AttentionConcat[:, 1, :, :]
        SplitAttention3 = AttentionConcat[:, 2, :, :]

        SplitAttention1 = torch.unsqueeze(SplitAttention1, dim=1)
        SplitAttention2 = torch.unsqueeze(SplitAttention2, dim=1)
        SplitAttention3 = torch.unsqueeze(SplitAttention3, dim=1)

        # Spatial Attention
        out1 = out1 + SplitAttention1 * out1
        out2 = out2 + SplitAttention2 * out2
        out3 = out3 + SplitAttention3 * out3

        out1 = F.avg_pool2d(out1, out1.size()[3])
        out2 = F.avg_pool2d(out2, out2.size()[3])
        out3 = F.avg_pool2d(out3, out3.size()[3])

        out1 = out1.view(out1.size(0), -1)
        out1 = F.normalize(out1, dim=1)

        out2 = out2.view(out2.size(0), -1)
        out2 = F.normalize(out2, dim=1)

        out3 = out3.view(out3.size(0), -1)
        out3 = F.normalize(out3, dim=1)

        all_feature = torch.cat([out1, out2, out3], dim=1)
        out = self.fc_class(all_feature)

        return out


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

def resnet32_ms(num_classes=10, include_top=False):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return MSNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50_ms(num_classes=10, include_top=False):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    return MSNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101_ms(num_classes=10, include_top=False):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return MSNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def get_resnet_ms(name='resnet32', num_classes=10):
    """
    Get back a resnet model tailored for cifar datasets and dimension which has been modified for a contrastive approach

    :param name: str
    :param num_classes: int
    :return: torch.nn.Module
    """
    name_to_resnet = {'resnet32': resnet32_ms,
                      'resnet50': resnet50_ms,
                      'resnet101': resnet101_ms,
                      "OriginRes50": resnet50,
                      "OriginRes34": resnet34}
    if name in name_to_resnet:
        return name_to_resnet[name](num_classes)
    else:
        raise ValueError('Model name %s not found'.format(name))

