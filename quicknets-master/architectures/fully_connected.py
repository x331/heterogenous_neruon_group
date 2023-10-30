import torch
import torch.nn as nn
from collections import OrderedDict



class EarlyExitFullyConnected(nn.Module):

    def __init__(self, layers, num_class=10, input_size=28*28):
        super().__init__()

        self.layers = []

        self.classifiers1 = []
        self.classifiers2 = []

        for i, layer in enumerate(layers):
            if i == 0:
                self.layers += [nn.Linear(input_size, layer)]
            else:
                self.layers += [nn.Linear(layers[i-1], layer)]

            self.layers += [nn.ReLU(inplace=True)]
            self.classifiers1 += [nn.Linear(layer, num_class)]
            self.classifiers2 += [nn.Linear(num_class, num_class)]

        self.phase = 0
        self.frozen_features = nn.Sequential()
        self.features = nn.Sequential(*self.layers[:2])
        self.classifier1 = nn.Sequential(*self.classifiers1[0:1])
        self.classifier2 = nn.Sequential(*self.classifiers2[0:1])


    def set_phase(self, phase):
        self.frozen_features = nn.Sequential(*self.layers[:phase*2])
        self.features = nn.Sequential(*self.layers[phase*2:phase*2+2])
        self.classifier1 = nn.Sequential(*self.classifiers1[phase:phase+1])
        self.classifier2 = nn.Sequential(*self.classifiers2[phase:phase+1])
        self.phase = phase

    def forward(self, x):
        output = self.frozen_features(x)
        output = self.features(output)
        output1 = self.classifier1(output)
        output2 = nn.functional.relu(output1)
        output2 = self.classifier2(output2)
        return output1, output2


def FC1():
    return EarlyExitFullyConnected([150, 100, 50])


class _Block(nn.Sequential):
    def __init__(self, layer, curr_input_size=28*28):
        super(_Block, self).__init__()

        self.add_module('linear%d' % 1, nn.Linear(layer, curr_input_size))
        self.add_module('relu%d'% 1,nn.ReLU(inplace=True))


class AdaptiveFullyConnected(nn.Module):

    def __init__(self, layer, num_class=10, input_size=28*28):
        super().__init__()

        self.layer = layer
        self.num_class = num_class
        self.features = nn.Sequential(OrderedDict([]))
        self.phase = 1
        self.features.add_module('block%d' % self.phase, _Block(input_size, layer))

        self.classifiers1 = []
        self.classifiers2 = []

        self.classifiers1 += [nn.Linear(layer, num_class)]
        self.classifiers2 += [nn.Linear(num_class + layer, num_class)]

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
        self.features.add_module('block%d' % self.phase, _Block(self.layer, self.layer))
        self.classifiers1 += [nn.Linear(self.layer, self.num_class)]
        self.classifiers2 += [nn.Linear(self.num_class + self.layer, self.num_class)]
        self.classifier1 = nn.Sequential(*self.classifiers1[self.phase - 1:self.phase])
        self.classifier2 = nn.Sequential(*self.classifiers2[self.phase - 1:self.phase])

    def forward(self, x):
        output = self.features(x)
        output1 = self.classifier1(output)
        output2 = torch.cat((output, output1), dim=1)
        output2 = self.classifier2(output2)
        return output1, output2
