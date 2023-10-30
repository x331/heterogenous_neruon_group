""
import torch
import torch.nn as nn
import torch.nn.functional as F

class block_conv(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, downsample=False, batchn=True):
        super(block_conv, self).__init__()
        self.downsample = downsample
        if downsample:
            self.down = psi(2)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if batchn:
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = identity()  # Identity

    def forward(self, x):
        if self.downsample:
            x = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

class auxillary_classifier(nn.Module):
    def __init__(self, avg_size=16, feature_size=256, input_features=256, in_size=32, num_classes=10, n_lin=0,
                 batchn=True):
        super(auxillary_classifier, self).__init__()
        self.n_lin = n_lin

        if n_lin == 0:
            feature_size = input_features

        self.blocks = []
        for n in range(self.n_lin):
            if n == 0:
                input_features = input_features
            else:
                input_features = feature_size

            if batchn:
                bn_temp = nn.BatchNorm2d(feature_size)
            else:
                bn_temp = identity()

            self.blocks.append(nn.Sequential(nn.Conv2d(input_features, feature_size,
                                                       kernel_size=3, stride=1, padding=1, bias=False), bn_temp))

        self.blocks = nn.ModuleList(self.blocks)
        if batchn:
            self.bn = nn.BatchNorm2d(feature_size)
        else:
            self.bn = identity()  # Identity

        self.avg_size = avg_size
        self.classifier = nn.Linear(feature_size * (in_size // avg_size) * (in_size // avg_size), num_classes)
        self.decision_layer = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        out = x
        for n in range(self.n_lin):
            out = self.blocks[n](out)
            out = F.relu(out)
        if (self.avg_size > 1):
            out = F.avg_pool2d(out, self.avg_size)
        out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out2 = self.decision_layer(F.relu(out))
        return out, out2


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, input):
        return input


class greedyNet(nn.Module):
    def __init__(self, block, num_blocks, feature_size=256, downsampling=1, downsample=[], batchnorm=True):
        super(greedyNet, self).__init__()
        self.in_planes = feature_size
        self.down_sampling = psi(downsampling)
        self.downsample_init = downsampling
        self.conv1 = nn.Conv2d(3 * downsampling * downsampling, self.in_planes, kernel_size=3, stride=1, padding=1,
                               bias=not batchnorm)

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(self.in_planes)
        else:
            self.bn1 = identity()  # Identity
        self.RELU = nn.ReLU()
        self.blocks = []
        self.block = block
        self.blocks.append(nn.Sequential(self.conv1, self.bn1, self.RELU))  # n=0
        self.batchn = batchnorm
        for n in range(num_blocks - 1):
            if n in downsample:
                pre_factor = 4
                self.blocks.append(
                    block(self.in_planes * pre_factor, self.in_planes * 2, downsample=True, batchn=batchnorm))
                self.in_planes = self.in_planes * 2
            else:
                self.blocks.append(block(self.in_planes, self.in_planes, batchn=batchnorm))

        self.blocks = nn.ModuleList(self.blocks)
        for n in range(num_blocks):
            for p in self.blocks[n].parameters():
                p.requires_grad = False

    def unfreezeGradient(self, n):
        for k in range(len(self.blocks)):
            for p in self.blocks[k].parameters():
                p.requires_grad = False

        for p in self.blocks[n].parameters():
            p.requires_grad = True

    def unfreezeAll(self):
        for k in range(len(self.blocks)):
            for p in self.blocks[k].parameters():
                p.requires_grad = True

    def add_block(self, downsample=False):
        if downsample:
            pre_factor = 4  # the old block needs this factor 4
            self.blocks.append(
                self.block(self.in_planes * pre_factor, self.in_planes * 2, downsample=True, batchn=self.batchn))
            self.in_planes = self.in_planes * 2
        else:
            self.blocks.append(self.block(self.in_planes, self.in_planes, batchn=self.batchn))

    def forward(self, a):
        x = a
        out = x
        if self.downsample_init > 1:
            out = self.down_sampling(x)
        for n in range(len(self.blocks)):
            out = self.blocks[n](out)
        return out