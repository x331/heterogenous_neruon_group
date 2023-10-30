import torch
import torch.nn as nn
from collections import OrderedDict


class _Block(nn.Sequential):
    def __init__(self, layer, curr_input_size=28):
        super(_Block, self).__init__()

        self.add_module('conv%d' % 1, nn.Conv2d(curr_input_size, layer, kernel_size=3, padding=1))
        self.add_module('relu%d'% 1,nn.ReLU(inplace=True))


def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1


class _InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(_InternalClassifier, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        input_size = int(input_size)
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels
        self.alpha = alpha

        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
            self.feature_size = output_channels*input_size*input_size
        else:
            red_input_size = int(input_size/red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
            self.forward = self.forward_w_pooling
            self.feature_size = output_channels*red_input_size*red_input_size

    def forward_w_pooling(self, x):
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        mixed = mixed.view(mixed.size(0), -1)
        return self.linear(mixed), mixed

    def forward_wo_pooling(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x), x


class _DecisionLayer(nn.Sequential):
    def __init__(self, input_size):
        super(_DecisionLayer, self).__init__()
        self.add_module('linear%d' % 1, nn.Linear(input_size, 256))
        self.add_module('relu%d' % 1, nn.ReLU(inplace=True))
        self.add_module('linear%d' % 2, nn.Linear(256, 1))


class AdaptiveConv(nn.Module):

    def __init__(self, layer, num_class=10, input_size=32):
        super().__init__()

        self.num_class = num_class
        self.feature_size = input_size
        self.features = nn.Sequential(OrderedDict([]))
        self.phase = 1
        self.layer = layer
        self.sub_phase = 0
        self.features.add_module('block%d_%d' % (self.phase, self.sub_phase), _Block(layer, 3))

        self.classifiers1 = []
        self.classifiers2 = []

        self.classifiers1 += [_InternalClassifier(self.feature_size, self.layer, self.num_class)]
        self.classifiers2 += [_DecisionLayer(self.num_class + self.classifiers1[-1].feature_size)]

        self.classifier1 = nn.Sequential(*self.classifiers1[self.phase - 1:self.phase])
        self.classifier2 = nn.Sequential(*self.classifiers2[self.phase - 1:self.phase])
        self.old_features = self.features

    def set_phase(self, phase):
        self.phase = phase
        self.old_features = self.features
        self.features = self.features[0:self.phase]
        self.classifier1 = nn.Sequential(*self.classifiers1[self.phase - 1:self.phase])
        self.classifier2 = nn.Sequential(*self.classifiers2[self.phase - 1:self.phase])

    def add_phase(self):
        self.phase += 1
        self.sub_phase = 0
        self.features.add_module('block%d_%d' % (self.phase, self.sub_phase), _Block(self.layer, self.layer))
        self.classifiers1 += [_InternalClassifier(self.feature_size, self.layer, self.num_class)]
        self.classifiers2 += [_DecisionLayer(self.num_class + self.classifiers1[-1].feature_size)]
        self.classifier1 = nn.Sequential(*self.classifiers1[self.phase - 1:self.phase])
        self.classifier2 = nn.Sequential(*self.classifiers2[self.phase - 1:self.phase])

    def add_block(self):
        self.sub_phase += 1
        self.features.add_module('block%d_%d' % (self.phase, self.sub_phase), _Block(self.layer, self.layer))

    def forward(self, x):
        output = self.features(x)
        output1, output2 = self.classifier1(output)
        output2 = torch.cat((output1, output2), dim=1)
        output2 = self.classifier2(output2)
        return output1, output2
