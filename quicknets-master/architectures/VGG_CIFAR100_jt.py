import torch
import torch.nn as nn
from collections import OrderedDict
from torch import Tensor
import torch.nn.functional as F

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}




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
        else:
            red_input_size = int(input_size/red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))


class _InternalClassifierTwoLayer(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, fc_params, alpha=0.5):
        super(_InternalClassifierTwoLayer, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        input_size = int(input_size)
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels
        self.alpha = alpha

        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels*input_size*input_size, int(fc_params[0]))
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear = nn.Linear(output_channels*red_input_size*red_input_size, int(fc_params[0]))
            self.forward = self.forward_w_pooling

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(int(fc_params[0]), int(fc_params[1]))
        self.linear3 = nn.Linear(int(fc_params[1]), num_classes)

    def forward_w_pooling(self, x):
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        out = self.relu(self.linear(mixed.view(mixed.size(0), -1)))
        out = self.dropout(out, 0.5)
        out = self.relu(self.linear2(out))
        out = self.dropout(out, 0.5)
        out = self.relu(self.linear3(out))
        return out

    def forward_wo_pooling(self, x):
        out = self.relu(self.linear(x.view(x.size(0), -1)))
        out = self.dropout(out, 0.5)
        out = self.relu(self.linear2(out))
        out = self.dropout(out, 0.5)
        out = self.relu(self.linear3(out))
        return out


class _FinalClassifier(nn.Module):
    def __init__(self, fc_params, input_size, input_channels, num_classes):
        super(_FinalClassifier, self).__init__()

        fc_layers = list()
        input_size = int(input_size)
        fc_layers.append(nn.Flatten())
        fc_layers.append(nn.Linear(input_size * input_size * input_channels, int(fc_params[0])))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))

        fc_layers.append(nn.Linear(int(fc_params[0]), int(fc_params[1])))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))

        fc_layers.append(nn.Linear(int(fc_params[1]), num_classes))
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd


class _Block(nn.Sequential):
    def __init__(self, layers, batch_norm=False, curr_input_size=32, channels=3):
        super(_Block, self).__init__()
        for i, layer in enumerate(layers):
            if layer == 'M':
                self.add_module('maxpool%d' % (i + 1), nn.MaxPool2d(kernel_size=2, stride=2))
                curr_input_size /= 2
                continue
            else:
                self.add_module('conv%d'%(i+1), nn.Conv2d(channels, layer, kernel_size=3, padding=1))
                if batch_norm:
                    self.add_module('batchnorm%d'%(i+1), nn.BatchNorm2d(layer))
                self.add_module('relu%d'%(i+1),nn.ReLU(inplace=True))
            channels = layer
        self.out_input_size = curr_input_size
        self.out_channels = channels


class VGGJT(nn.Module):

    def __init__(self, num_classes=10, batch_norm=False, config=cfg['D'],
                 output_layers=[0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1], fc_params=[512, 512],
                 two_layer_ic = False):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([]))
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        self.config = config
        self.output_layers = output_layers
        self.fc_params = fc_params
        self.phase = 1
        self.internal_classifier = _InternalClassifierTwoLayer if two_layer_ic else _InternalClassifier

        blocks, input_size, channels = self.get_blocks(self.phase, self.config, self.output_layers, self.batch_norm, 32, 3)
        for i, block in enumerate(blocks):
            self.features.add_module('block%d' % (i+1), block)
        if self.phase == sum(self.output_layers):
            self.classifiers = [_FinalClassifier(self.fc_params, input_size, channels, self.num_classes)]
        else:
            self.classifiers = [self.internal_classifier(input_size, channels, self.num_classes)]
        self.classifier = nn.Sequential(*self.classifiers[self.phase - 1:self.phase])

    @staticmethod
    def get_blocks(phase, layers, output_layers, batch_norm=False, input_size=32, input_channels=3):
        blocks = []
        count = 0
        for i in range(phase):
            block_start = count
            for o in output_layers[count:]:
                count += 1
                if o == 1:
                    break
            block = _Block(layers[block_start:count], batch_norm, input_size, input_channels)
            input_size = block.out_input_size
            input_channels = block.out_channels
            blocks.append(block)

        return blocks, input_size, input_channels

    def add_phase(self):
        self.phase += 1
        blocks, input_size, channels = self.get_blocks(self.phase, self.config, self.output_layers, self.batch_norm, 32,
                                                       3)
        self.features.add_module('block%d' % self.phase, blocks[-1])

        if self.phase == sum(self.output_layers):
            self.classifiers.append(_FinalClassifier(self.fc_params, input_size, channels, self.num_classes))
        else:
            self.classifiers.append(self.internal_classifier(input_size, channels, self.num_classes))
        self.classifier = nn.Sequential(*self.classifiers[self.phase - 1:self.phase])

    def set_phase(self, phase):
        self.phase = phase
        self.old_features = self.features
        self.features = self.features[0:self.phase]
        self.classifier = nn.Sequential(*self.classifiers[self.phase - 1:self.phase])

    def forward(self, x):

        output = self.features(x)
        output = self.classifier(output)
        return output
