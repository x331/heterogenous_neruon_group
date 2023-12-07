'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math

from .configs import InfoPro, InfoPro_balanced_memory,h_split_ratios
from .auxiliary_nets import Decoder, AuxClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0,split=False, beginning=False,h_ratio=.5):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate)
        self.beginning = beginning
        self.h_ratio = h_ratio
        self.split = split
        self.inplanes = inplanes
        self.planes = planes
        if not split :
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        else:
            if beginning:
                out_chan1 = math.floor(planes*h_ratio)
                out_chan2 = planes - out_chan1
                # print(1,out_chan1,out_chan2)
                self.conv1a = conv3x3(inplanes, out_chan1, stride)
                self.conv1b = conv3x3(inplanes, out_chan2, stride) 
            else:
                in_chan1 = math.floor(inplanes*h_ratio)
                in_chan2 = inplanes - in_chan1
                out_chan1 = math.floor(planes*h_ratio)
                out_chan2 = planes - out_chan1
                # print(2,out_chan1,out_chan2)
                self.conv1a = conv3x3(in_chan1, out_chan1, stride)
                self.conv1b = conv3x3(in_chan2, out_chan2, stride)    
            num_chan1 = math.floor(planes*h_ratio)
            num_chan2 = planes - num_chan1
            # print(4,num_chan1,num_chan2)
            self.conv2a = conv3x3(num_chan1, num_chan1, stride)
            self.conv2b = conv3x3(num_chan2, num_chan2, stride)    
            if  downsample != None:
                if self.beginning:
                    # print(downsample,self.expansion)
                    self.downsample = nn.Sequential(
                        nn.Conv2d(inplanes, planes * downsample, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes * downsample)
                    )
                else:
                    in_chan1 = math.floor(inplanes*h_ratio)
                    in_chan2 = inplanes - in_chan1
                    out_chan1 = math.floor(planes*h_ratio)
                    out_chan2 = planes - out_chan1 
                    # print(3,out_chan1,out_chan2,flush=True)
                    self.downsample = [nn.Sequential(
                            nn.Conv2d(in_chan1, out_chan1 * downsample, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_chan1 * downsample)
                        ),
                        nn.Sequential(
                            nn.Conv2d(in_chan2, out_chan2 * downsample, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_chan2 * downsample)
                        )] 

    def forward(self, x):
        residual = x
        
        out = 0
        if not self.split:
            out = self.conv1(x)
        else:
            # xa = self.conv1a(x)
            # xb = self.conv1b(x)
            # out = torch.cat((xa,xb),dim=1)
            if self.beginning:
                xa = self.conv1a(x)
                xb = self.conv1b(x)
                out = torch.cat((xa,xb),dim=1)
            else:
                in_chan1 = math.floor(self.inplanes*self.h_ratio)
                in_chan2 = self.inplanes - in_chan1
                xa = self.conv1a(x[:,:in_chan1,:,:])
                xb = self.conv1b(x[:,in_chan1:,:,:])
                out = torch.cat((xa,xb),dim=1)
        
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        if not self.split:
            out = self.conv2(out)
        else:
            # xa = self.conv2a(out)
            # xb = self.conv2b(out)
            # out = torch.cat((xa,xb),dim=1)
            in_chan1 = math.floor(self.planes*self.h_ratio)
            in_chan2 = self.planes - in_chan1
            xa = self.conv2a(out[:,:in_chan1,:,:])
            xb = self.conv2b(out[:,in_chan1:,:,:])
            out = torch.cat((xa,xb),dim=1)         
            
        out = self.bn2(out)

        residual = 0
        if self.downsample is not None:
            if not self.split:
                residual = self.downsample(x)
            else:
                if self.beginning:
                    residual = self.downsample(x)
                else:
                    in_chan1 = math.floor(self.inplanes*self.h_ratio)
                    xa = self.downsample[0](x[:,:in_chan1,:,:])
                    xb = self.downsample[1](x[:,in_chan1:,:,:])
                    residual = torch.cat(xa,xb,dim=1)
                    # residual = self.downsample(x)
                
        print(out.shape,residual.shape if type(residual) != 'int' else residual)
        out += residual
        out = self.relu(out)

        return out
    
    
# class BasicBlock_h(nn.Module):
#     expansion=1
#     def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0,beginning=False,h_ratio=1):
#         super(BasicBlock_h, self).__init__()
#         if beginning:
#             out_chan1 = math.floor(planes*h_ratio)
#             out_chan2 = planes - out_chan1
#             self.conv1a = conv3x3(inplanes, out_chan1, stride)
#             self.conv1a = conv3x3(inplanes, out_chan2, stride) 
#         else:
#             in_chan1 = math.floor(inplanes*h_ratio)
#             in_chan2 = inplanes - in_chan1
#             out_chan1 = math.floor(planes*h_ratio)
#             out_chan2 = inplanes - out_chan1
#             self.conv1a = conv3x3(in_chan1, out_chan1, stride)
#             self.conv1a = conv3x3(in_chan2, out_chan2, stride)    
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)

#         num_chan1 = math.floor(planes*h_ratio)
#         num_chan2 = inplanes - out_chan1
#         self.conv2a = conv3x3(num_chan1, num_chan1, stride)
#         self.conv2b = conv3x3(num_chan2, num_chan2, stride)    
        
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.dropout = nn.Dropout(p=dropout_rate)

#     def forward(self, x):
#         residual = x
        
#         xa = self.conv1a(img)
#         xb = self.conv1b(img)
#         out = torch.cat((xa,xb),dim=1)
        
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.dropout(out)
        
#         xa = self.conv2a(out)
#         xb = self.conv2b(out)
#         out = torch.cat((xa,xb),dim=1)
        
#         out = self.bn2(out)
        
#         if self.downsample is not None:
#             # xa = self.downsample[0](x)
#             # xb = self.downsample[1](x)
#             # residual = torch.cat(xa,xb,dim=1)
#             residual = self.downsample(x)

            
#         out += residual
#         out = self.relu(out)
#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class InfoProResNet(nn.Module):

    def __init__(self, block, layers, arch, local_module_num, batch_size, image_size=32,
                 balanced_memory=False, dataset='cifar10', class_num=10,
                 wide_list=(16, 16, 32, 64), dropout_rate=0,
                 aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128,
                 train_type='locally',
                 loss_type='class', info_class_ratio=.0,
                 h_split = -1):
        super(InfoProResNet, self).__init__()

        assert arch in ['resnet20','resnet32', 'resnet110'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."
        
        try:
            self.infopro_config = InfoPro_balanced_memory[arch][dataset][local_module_num] \
                if balanced_memory else InfoPro[arch][local_module_num]
        except:
            raise NotImplementedError  
                                                  
        try:
            self.h_split_ratios = h_split_ratios[arch][local_module_num][h_split]
        except:
            raise NotImplementedError

        self.inplanes = wide_list[0]
        self.dropout_rate = dropout_rate
        self.feature_num = wide_list[-1]
        self.class_num = class_num
        self.local_module_num = local_module_num
        self.layers = layers
        self.train_type = train_type
        self.loss_type = loss_type
        self.info_class_ratio = info_class_ratio
        self.h_split = h_split
        
        if self.h_split == -1:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            num_chan1 = math.floor(self.inplanes*self.h_split_ratios[0])
            num_chan2 = self.inplanes-num_chan1
            self.conv1a = nn.Conv2d(3, num_chan1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1b = nn.Conv2d(3, num_chan2, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, wide_list[1], layers[0],stage=1)
        self.layer2 = self._make_layer(block, wide_list[2], layers[1], stride=2,stage=2)
        self.layer3 = self._make_layer(block, wide_list[3], layers[2], stride=2,stage=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_num, self.class_num)
        

        self.criterion_ce = nn.CrossEntropyLoss()        
        
        if self.h_split == -1:
            for item in self.infopro_config:
                module_index, layer_index = item
                if self.loss_type == 'info' or self.loss_type == 'both':
                    exec('self.decoder_' + str(module_index) + '_' + str(layer_index) +
                        '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')
                    exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
                        '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                        'loss_mode=local_loss_mode, class_num=class_num, '
                        'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
                if self.loss_type == 'class'  or self.loss_type == 'both':
                    exec('self.pred_head_' + str(module_index) + '_' + str(layer_index) +
                        '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                        'loss_mode=local_loss_mode, class_num=class_num, '
                        'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
        else:
            # for i,item in enumerate(self.infopro_config):
            #     for group in ['a','b']:
            #         module_index, layer_index = item
            #         chan = math.floor(wide_list[module_index]*self.h_split_ratios[i])
            #         if group== 'b':
            #             chan = wide_list[module_index]-chan
                        
            #         if self.loss_type == 'info' or self.loss_type == 'both':
            #             exec('self.decoder_' + str(module_index) + '_' + str(layer_index) + '_' + group +
            #                 '= Decoder(' + str(chan) + ', image_size, widen=aux_net_widen)')
            #             exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) + '_' + group +
            #                 '= AuxClassifier(' + str(chan) + ', net_config=aux_net_config, '
            #                 'loss_mode=local_loss_mode, class_num=class_num, '
            #                 'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
            #         if self.loss_type == 'class'  or self.loss_type == 'both':
            #             exec('self.pred_head_' + str(module_index) + '_' + str(layer_index) + '_' + group +
            #                 '= AuxClassifier(' + str(chan) + ', net_config=aux_net_config, '
            #                 'loss_mode=local_loss_mode, class_num=class_num, '
            #                 'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
            # for i,item in enumerate(self.infopro_config):
            #         module_index, layer_index = item
            #         chan_a = math.floor(wide_list[module_index]*self.h_split_ratios[i])
            #         chan_b = wide_list[module_index]-chan_a
                        
            #         if self.loss_type == 'info' or self.loss_type == 'both':
            #             exec('self.decoder_' + str(module_index) + '_' + str(layer_index) +
            #                 '= Decoder(' + str(chan_a) + ', image_size, widen=aux_net_widen)')
            #             exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
            #                 '= AuxClassifier(' + str(chan_a) + ', net_config=aux_net_config, '
            #                 'loss_mode=local_loss_mode, class_num=class_num, '
            #                 'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
            #         if self.loss_type == 'class'  or self.loss_type == 'both':
            #             exec('self.pred_head_' + str(module_index) + '_' + str(layer_index) +
            #                 '= AuxClassifier(' + str(chan_b) + ', net_config=aux_net_config, '
            #                 'loss_mode=local_loss_mode, class_num=class_num, '
            #                 'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
            for i,item in enumerate(self.infopro_config):
                    module_index, layer_index = item                        
                    if self.loss_type == 'info' or self.loss_type == 'both':
                        exec('self.decoder_' + str(module_index) + '_' + str(layer_index) +
                            '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')
                        exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
                            '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                            'loss_mode=local_loss_mode, class_num=class_num, '
                            'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
                    if self.loss_type == 'class'  or self.loss_type == 'both':
                        exec('self.pred_head_' + str(module_index) + '_' + str(layer_index) +
                            '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                            'loss_mode=local_loss_mode, class_num=class_num, '
                            'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # print( m.kernel_size[0] , m.kernel_size[1] , m.out_channels,m)
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if 'cifar' in dataset:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)

    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]
               
    
    def _get_local_mod_boundaries(self, stage):
        boundaries = []
        for item in self.infopro_config:
            module_index, layer_index = item
            if module_index == stage:
                boundaries.append(layer_index)
        return boundaries
    
    def _get_local_mod_pos(self, stage,layer):
        pos = 0
        for item in self.infopro_config:
            module_index, layer_index = item
            if module_index == stage and layer_index >= layer :
                break
            pos+=1
        return pos

    # def _make_layer(self, block, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion)
    #         )
            
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

    #     return nn.Sequential(*layers)
    
    
    def _make_layer(self, block, planes, blocks, stride=1, stage=None):

        if self.h_split != -1:
            first_conv_groups = False
            if (0 in self._get_local_mod_boundaries(stage-1) if stage == 1 else (self.layers[stage-2]-1) in self._get_local_mod_boundaries(stage-1)):
                first_conv_groups = True

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.h_split == -1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )
            else:
                # downsample = nn.Sequential(
                #     nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                #     nn.BatchNorm2d(planes * block.expansion)
                # )
                downsample = block.expansion
            
        layers = []
        if self.h_split == -1:
            layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))
        else:   
            layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate,split=True,beginning=first_conv_groups,h_ratio=self.h_split_ratios[self._get_local_mod_pos(stage-1,0)]))
            self.inplanes = planes * block.expansion
            for b in range(1, blocks):
                first_conv_groups = False
                if (b-1) in self._get_local_mod_boundaries(stage):
                    first_conv_groups = True
                layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate,split=True, beginning=first_conv_groups,h_ratio=self.h_split_ratios[self._get_local_mod_pos(stage-1,0)]))
                

        return nn.Sequential(*layers)

    def forward_original(self, img):
        x = 0
        if self.h_split == -1:
            x = self.conv1(img)
        else:
            xa = self.conv1a(img)
            xb = self.conv1b(img)
            x = torch.cat((xa,xb),dim=1)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, img, target=None,
                ixx_1=0, ixy_1=0,
                ixx_2=0, ixy_2=0,
                no_early_exit_pred = False,
                target_module=None):

        if self.training or not no_early_exit_pred:            
            stage_i = 0
            layer_i = 0
            local_module_i = 0
            loss_per_exit = []
            pred_per_exit = []

            x=0
            if self.h_split == -1:
                x = self.conv1(img)
            else:
                xa = self.conv1a(img)
                xb = self.conv1b(img)
                x = torch.cat((xa,xb),dim=1)
            x = self.bn1(x)
            x = self.relu(x)
            
            if self.h_split == -1:
                if local_module_i <= self.local_module_num - 2:
                    if self.infopro_config[local_module_i][0] == stage_i \
                            and self.infopro_config[local_module_i][1] == layer_i:
                        if self.train_type == 'local':
                            if self.loss_type in ['class','info','both']:
                                infoproloss = 0 
                                classloss = 0
                                preds = 0
                                loss = 0
                                if self.loss_type != 'class':
                                    ratio = local_module_i / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                                    ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                                    ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                                    loss_ixx = eval('self.decoder_' + str(stage_i) + '_' + str(layer_i))(x, self._image_restore(img))
                                    loss_ixy,preds = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(x, target)
                                    infoproloss = ixx_r * loss_ixx + ixy_r * loss_ixy
                                if self.loss_type != 'info':
                                    classloss, preds = eval('self.pred_head_' + str(stage_i) + '_' + str(layer_i))(x, target)
                                    
                                if self.loss_type == 'class':
                                    loss = classloss
                                elif self.loss_type == 'info':
                                    loss = infoproloss
                                elif self.loss_type == 'both':
                                    loss =  infoproloss*(self.info_class_ratio)+classloss*(1-self.info_class_ratio)
                                    
                                if self.training :    
                                    loss.backward()        
                                loss_per_exit.append(loss)
                                pred_per_exit.append(preds)
                                x = x.detach()

                        else:
                            loss_ixy, preds = eval('self.pred_head_' + str(stage_i) + '_' + str(layer_i))(x, target)
                            loss_per_exit.append(loss_ixy)
                            
                            if self.train_type == 'layer':
                                # means we reached the classifier of the target module
                                if target_module == local_module_i:
                                    pred_per_exit.append(preds)
                                    for _ in range(self.local_module_num - 1 - local_module_i):
                                        loss_per_exit.append(torch.zeros_like(loss_ixy))
                                        pred_per_exit.append(torch.zeros_like(preds))
                                        
                                    if self.training:
                                        loss_ixy.backward()
                                    return pred_per_exit, loss_per_exit
                                else:
                                    # detach the current module from computation graph, only need to keep the target module
                                    x = x.detach()
                        
                            pred_per_exit.append(preds)
                        local_module_i += 1  
                        
            else:
                if local_module_i <= self.local_module_num - 2:
                    if self.infopro_config[local_module_i][0] == stage_i \
                            and self.infopro_config[local_module_i][1] == layer_i:
                        
                        chan_1 = math.floor(self.h_split_ratios[self._get_local_mod_pos(stage_i,layer_i)]*x.shape[1])
                        xa_a = x[:,:chan_1,:,:]
                        xb_a = x[:,chan_1:,:,:]
                        xa_d = xa_a.detach().clone()
                        xb_d = xb_a.detach().clone()
                        xa = torch.cat((xa_a,xb_d),dim=1)
                        xb = torch.cat((xa_d,xb_a),dim=1)

                        if self.train_type == 'local':
                            if self.loss_type in ['class','info','both']:
                                infoproloss = 0 
                                classloss = 0
                                preds = 0
                                loss = 0
                                if self.loss_type != 'class':
                                    ratio = local_module_i / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                                    ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                                    ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                                    loss_ixx = eval('self.decoder_' + str(stage_i) + '_' + str(layer_i))(xa, self._image_restore(img))
                                    loss_ixy,preds = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(xa, target)
                                    infoproloss = ixx_r * loss_ixx + ixy_r * loss_ixy
                                if self.loss_type != 'info':
                                    classloss, preds = eval('self.pred_head_' + str(stage_i) + '_' + str(layer_i))(xb, target)
                                    
                                if self.loss_type == 'class':
                                    print('not supported')
                                elif self.loss_type == 'info':
                                    print('not supported')
                                elif self.loss_type == 'both':
                                    # loss =  infoproloss*(self.info_class_ratio)+classloss*(1-self.info_class_ratio)
                                    loss =  [infoproloss,classloss]

                                if self.training : 
                                    for l in loss:
                                        l.backward(retain_graph=True)  
                                loss_per_exit.append(loss)
                                pred_per_exit.append(preds)
                                x = x.detach()

                        else:
                            print('not supported')
                            # loss_ixy, preds = eval('self.pred_head_' + str(stage_i) + '_' + str(layer_i))(xb, target)
                            # loss_per_exit.append(loss_ixy)
                            
                            # if self.train_type == 'layer':
                            #     # means we reached the classifier of the target module
                            #     if target_module == local_module_i:
                            #         pred_per_exit.append(preds)
                            #         for _ in range(self.local_module_num - 1 - local_module_i):
                            #             loss_per_exit.append(torch.zeros_like(loss_ixy))
                            #             pred_per_exit.append(torch.zeros_like(preds))
                                        
                            #         if self.training:
                            #             loss_ixy.backward()
                            #         return pred_per_exit, loss_per_exit
                            #     else:
                            #         # detach the current module from computation graph, only need to keep the target module
                            #         x = x.detach()
                        
                            # pred_per_exit.append(preds)
                        local_module_i += 1                   
                        
            for stage_i in (1, 2, 3):
                for layer_i in range(self.layers[stage_i - 1]):

                    print(stage_i,layer_i,x.shape) 
                    x = eval('self.layer' + str(stage_i))[layer_i](x)

                    if self.h_split == -1:
                        if local_module_i <= self.local_module_num - 2:
                            if self.infopro_config[local_module_i][0] == stage_i \
                                    and self.infopro_config[local_module_i][1] == layer_i:
        
                                if self.train_type == 'local':
                                    if self.loss_type in ['class','info','both']:
                                        infoproloss = 0 
                                        classloss = 0
                                        preds = 0
                                        loss = 0
                                        if self.loss_type != 'class':
                                            ratio = local_module_i / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                                            ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                                            ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                                            loss_ixx = eval('self.decoder_' + str(stage_i) + '_' + str(layer_i))(x, self._image_restore(img))
                                            loss_ixy,preds = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(x, target)
                                            infoproloss = ixx_r * loss_ixx + ixy_r * loss_ixy
                                        if self.loss_type != 'info':
                                            classloss, preds = eval('self.pred_head_' + str(stage_i) + '_' + str(layer_i))(x, target)
                                            
                                        if self.loss_type == 'class':
                                            loss = classloss
                                        elif self.loss_type == 'info':
                                            loss = infoproloss
                                        elif self.loss_type == 'both':
                                            loss =  infoproloss*(self.info_class_ratio)+classloss*(1-self.info_class_ratio)
                                            
                                        if self.training : 
                                            loss.backward()     
                                        loss_per_exit.append(loss)
                                        pred_per_exit.append(preds)
                                        x = x.detach()
                                                

                                    
                                else:
                                    loss_ixy, preds = eval('self.pred_head_' + str(stage_i) + '_' + str(layer_i))(x, target)
                                    loss_per_exit.append(loss_ixy)
                                    
                                    if self.train_type == 'layer':
                                        # means we reached the classifier of the target module
                                        if target_module == local_module_i:
                                            pred_per_exit.append(preds)
                                            for _ in range(self.local_module_num - 1 - local_module_i):
                                                loss_per_exit.append(torch.zeros_like(loss_ixy))
                                                pred_per_exit.append(torch.zeros_like(preds))
                                            if self.training:
                                                loss_ixy.backward()
                                            return pred_per_exit, loss_per_exit
                                        else:
                                            # detach the current module from computation graph, only need to keep the target module
                                            x = x.detach()
                                
                                    pred_per_exit.append(preds)
                        
                                local_module_i += 1

                    else:
                        if local_module_i <= self.local_module_num - 2:
                            if self.infopro_config[local_module_i][0] == stage_i \
                                    and self.infopro_config[local_module_i][1] == layer_i:
                                
                                chan_1 = math.floor(self.h_split_ratios[self._get_local_mod_pos(stage_i,layer_i)]*x.shape[1])
                                xa_a = x[:,:chan_1,:,:]
                                xb_a = x[:,chan_1:,:,:]
                                xa_d = xa_a.detach().clone()
                                xb_d = xb_a.detach().clone()
                                xa = torch.cat((xa_a,xb_d),dim=1)
                                xb = torch.cat((xa_d,xb_a),dim=1)

                                if self.train_type == 'local':
                                    if self.loss_type in ['class','info','both']:
                                        infoproloss = 0 
                                        classloss = 0
                                        preds = 0
                                        loss = 0
                                        if self.loss_type != 'class':
                                            ratio = local_module_i / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                                            ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                                            ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                                            loss_ixx = eval('self.decoder_' + str(stage_i) + '_' + str(layer_i))(xa, self._image_restore(img))
                                            loss_ixy,preds = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(xa, target)
                                            infoproloss = ixx_r * loss_ixx + ixy_r * loss_ixy
                                        if self.loss_type != 'info':
                                            classloss, preds = eval('self.pred_head_' + str(stage_i) + '_' + str(layer_i))(xb, target)
                                            
                                        if self.loss_type == 'class':
                                            print('not supported')
                                        elif self.loss_type == 'info':
                                            print('not supported')
                                        elif self.loss_type == 'both':
                                            loss =  [infoproloss,classloss]
                                            
                                        if self.training : 
                                            for l in loss:
                                                l.backward(retain_graph=True)  
                                        loss_per_exit.append(loss)
                                        pred_per_exit.append(preds)
                                        x = x.detach()
                                                

                                    
                                else:
                                    print('not supported')
                                    # loss_ixy, preds = eval('self.pred_head_' + str(stage_i) + '_' + str(layer_i))(x, target)
                                    # loss_per_exit.append(loss_ixy)
                                    
                                    # if self.train_type == 'layer':
                                    #     # means we reached the classifier of the target module
                                    #     if target_module == local_module_i:
                                    #         pred_per_exit.append(preds)
                                    #         for _ in range(self.local_module_num - 1 - local_module_i):
                                    #             loss_per_exit.append(torch.zeros_like(loss_ixy))
                                    #             pred_per_exit.append(torch.zeros_like(preds))
                                    #         if self.training:
                                    #             loss_ixy.backward()
                                    #         return pred_per_exit, loss_per_exit
                                    #     else:
                                    #         # detach the current module from computation graph, only need to keep the target module
                                    #         x = x.detach()
                                
                                    # pred_per_exit.append(preds)
                        
                                local_module_i += 1                                
                            

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.fc(x)
            pred_per_exit.append(logits)
            fc_loss = self.criterion_ce(logits, target)
            loss_per_exit.append(fc_loss)
            if self.train_type == 'local':
                loss = fc_loss
                if self.training:
                    loss.backward()            
            return pred_per_exit, loss_per_exit

        else:
            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.fc(x)
            loss = self.criterion_ce(logits, target)
            return [logits], [loss]


def resnet8(**kwargs):
    model = InfoProResNet(BasicBlock, [1, 1, 1], arch='resnet20', **kwargs)
    return model

def resnet14(**kwargs):
    model = InfoProResNet(BasicBlock, [2, 2, 2], arch='resnet20', **kwargs)
    return model

def resnet20(**kwargs):
    model = InfoProResNet(BasicBlock, [3, 3, 3], arch='resnet20', **kwargs)
    return model


def resnet32(**kwargs):
    model = InfoProResNet(BasicBlock, [5, 5, 5], arch='resnet32', **kwargs)
    return model


def resnet44(**kwargs):
    model = InfoProResNet(BasicBlock, [7, 7, 7], arch='resnet44', **kwargs)
    return model


def resnet56(**kwargs):
    model = InfoProResNet(BasicBlock, [9, 9, 9], arch='resnet56', **kwargs)
    return model


def resnet110(**kwargs):
    model = InfoProResNet(BasicBlock, [18, 18, 18], arch='resnet110', **kwargs)
    return model


def resnet1202(**kwargs):
    model = InfoProResNet(BasicBlock, [200, 200, 200], arch='resnet1202', **kwargs)
    return model


def resnet164(**kwargs):
    model = InfoProResNet(Bottleneck, [18, 18, 18], arch='resnet164', **kwargs)
    return model


def resnet1001(**kwargs):
    model = InfoProResNet(Bottleneck, [111, 111, 111], arch='resnet1001', **kwargs)
    return model



# if __name__ == '__main__':
#     net = resnet32(local_module_num=16, batch_size=256, image_size=32,
#                  balanced_memory=False, dataset='cifar10', class_num=10,
#                  wide_list=(16, 16, 32, 64), dropout_rate=0,
#                  aux_net_config='1c2f', local_loss_mode='contrast',
#                  aux_net_widen=1, aux_net_feature_dim=128)
#     y = net(torch.randn(4, 3, 32, 32), torch.zeros(4).long())

    # print(net)
    # print(y.size())