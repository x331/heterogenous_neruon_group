import torch
from torch import nn, Tensor
from typing import Dict, Iterable, Callable
import argparse
import random
from torchvision import transforms
import os
import sys
sys.path.append('../')
from datasets import IndexedCIFAR10
sys.path.append('../../')
from architectures.conv import AdaptiveConv


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


def save_clustering_data(seed=0, batch_size=128, epochs=100, dataset='CIFAR10', block=1, decision_layer=False, phase=1):
    num_classes = 10  # todo change for other datasets
    params = [dataset, seed, batch_size, epochs, block, decision_layer]
    model_name = '_'.join([str(x) for x in params])

    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)

    if dataset == 'CIFAR10':
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        training = IndexedCIFAR10(root='../../data',
                                  transform=transforms.Compose([transforms.RandomCrop(size=(32, 32), padding=4),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=mean,
                                                                                     std=std)]),
                                  train=True,
                                  download=True)
        train_loader = torch.utils.data.DataLoader(dataset=training, batch_size=batch_size, shuffle=True)

    # Test if GPU is available
    gpu = torch.cuda.is_available()
    net = torch.load(os.path.join('.', 'models', model_name + '_exit' + str(phase)))
    # todo: add support for loading a network with different phase then setting the phase

    net.set_phase(phase)
    fe = FeatureExtractor(net, layers=['classifier1'])

    activations = torch.zeros([len(train_loader.dataset), 1024])  # todo: generalize this
    outputs = torch.zeros([len(train_loader.dataset), 10])
    correct = torch.zeros(len(train_loader.dataset), dtype=torch.bool)
    labels = torch.zeros(len(train_loader.dataset), dtype=torch.long)

    for i, (x, y, indices) in enumerate(train_loader):

        if gpu:
            x = x.cuda()

        # loss calculation and gradient update:
        features = fe.forward(x)
        labels[indices] = y
        o = features['classifier1'][0]
        pred = o.max(1)[1]  # get the index of the max log-probability
        correct_array = pred.eq(y.cuda())
        correct[indices] = correct_array.detach().cpu()
        outputs[indices] = o.detach().cpu()
        activations[indices] = features['classifier1'][1].detach().cpu()

    torch.save(activations, os.path.join('.', 'results', model_name + '_exit' + str(phase) + '_activations'))
    torch.save(outputs, os.path.join('.', 'results', model_name + '_exit' + str(phase) + '_outputs'))
    torch.save(correct, os.path.join('.', 'results', model_name + '_exit' + str(phase) + '_correct'))
    torch.save(labels, os.path.join('.', 'results', model_name + '_exit' + str(phase) + '_labels'))



if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
    parser.add_argument('--epochs', help='max epochs for each phase', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--block', type=int, default=1, help="max blocks to train")
    parser.add_argument('--decision_layer', dest='decision_layer', action='store_true', default=False,
                        help="train decision layer")
    parser.add_argument('--phase', help='phase', type=int, default=1)

    args = parser.parse_args()

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    save_clustering_data(**args)

    print()

# checklist for adding new param:
# 1. Add to parser
# 2. Add to function definition
# 3. Add to model name