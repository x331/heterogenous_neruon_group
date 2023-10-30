import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import sys
import random
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler
sys.path.append('../')
from datasets import IndexedMNIST, IndexedFashionMNIST
from train_utils import train_decision_layer, train_epoch
from torchvision import datasets, transforms
sys.path.append('../../')
from architectures.fully_connected import AdaptiveFullyConnected


def get_decision_layer_loader(train_loader, batch_size, net, gpu, untrained_sample_indices, dataset):
    net.eval()
    trained_indices = []
    correct = 0
    total = 0

    for i, (x, y, indices) in enumerate(train_loader):

        if gpu:
            x = x.cuda()
            y = y.cuda()

        # loss calculation and gradient update:
        outputs1, outputs2 = net.forward(x)
        pred = outputs1.data.max(1)[1]  # get the index of the max log-probability
        correct_array = pred.eq(y)
        correct += torch.sum(correct_array)
        total += len(y)
        trained_indices += indices[correct_array].tolist()

    net.train()

    trained_indices = list(set(trained_indices))
    if len(untrained_sample_indices)-len(trained_indices) == 0:
        return None, True
    weights = torch.zeros(len(train_loader.dataset))
    weights[untrained_sample_indices] = 1/(len(untrained_sample_indices)-len(trained_indices))
    weights[trained_indices] = 1/(len(trained_indices))

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                         sampler=WeightedRandomSampler(weights.type('torch.DoubleTensor'), 60000, replacement=True))

    # loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
    #                                      sampler=SubsetRandomSampler(np.setdiff1d(untrained_sample_indices, trained_indices)))

    return loader, False


def get_train_loader(batch_size, untrained_sample_indices, dataset, num_classes=10):
    labels = torch.tensor(dataset.data.targets)
    untrained_class_indices = []
    class_samples = []
    print('-------------------------------------------------------')
    print('remaining samples per class:')
    for i in range(num_classes):
        class_indices = (labels == i).int().nonzero(as_tuple=True)[0]
        untrained_class_indices.append(np.intersect1d(class_indices, untrained_sample_indices))
        print('class %d:' %(i+1), len(untrained_class_indices[-1]))
        class_samples.append(len(untrained_class_indices[-1]))

    weights = torch.zeros(len(dataset))
    for i, c in enumerate(class_samples):
        if c == 0:
            continue
        weights[untrained_class_indices[i]] = 1 / c

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                             sampler=WeightedRandomSampler(weights, len(weights)))
    return loader

def train(seed=0, batch_size=128, save_params=False, epochs=100, method='layer_wise',dataset='MNIST'):

    params = [dataset, seed, batch_size, epochs, method]
    model_name = '_'.join([str(x) for x in params])

    # Create directories
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists('models'):
        os.makedirs('models')

    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Create dataloaders
    if dataset == 'MNIST':
        training = IndexedMNIST(root='../../data', transform=transforms.Compose([transforms.ToTensor(),
                                                                             transforms.Lambda(lambda x: x.view(-1))]),
                                train=True,
                                download=True)
        train_loader = torch.utils.data.DataLoader(dataset=training, batch_size=batch_size, shuffle=True)

        testing = datasets.MNIST(root='../../data', transform=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Lambda(lambda x: x.view(-1))]),
                                 train=False,
                                 download=True)
        test_loader = torch.utils.data.DataLoader(dataset=testing, batch_size=batch_size, shuffle=True)
    elif dataset == 'FashionMNIST':
        training = IndexedFashionMNIST(root='../../data', transform=transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.RandomAffine(degrees=7,
                                                                                                         translate=(
                                                                                                         0.05,
                                                                                                         0.05),
                                                                                                         scale=(
                                                                                                         0.9, 1.1)),
                                                                                 transforms.Lambda(
                                                                                     lambda x: x.view(-1))]),
                                train=True,
                                download=True)
        train_loader = torch.utils.data.DataLoader(dataset=training, batch_size=batch_size, shuffle=True)

        testing = datasets.FashionMNIST(root='../../data', transform=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Lambda(
                                                                                      lambda x: x.view(-1))]),
                                 train=False,
                                 download=True)
        test_loader = torch.utils.data.DataLoader(dataset=testing, batch_size=batch_size, shuffle=True)

    # Create network
    net = AdaptiveFullyConnected(150)

    # Test if GPU is available
    gpu = torch.cuda.is_available()
    if gpu:
        print("CUDA available")
        net = net.cuda()

    optimizer = torch.optim.Adam([{'params': net.features[-1].parameters()},
                                  {'params': net.classifier1.parameters()}], lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)

    loss_metric = nn.CrossEntropyLoss()
    loss_metric2 = nn.BCEWithLogitsLoss()

    phase = 1
    untrained_sample_indices = torch.arange(len(train_loader.dataset))
    training_samples = [len(untrained_sample_indices)]
    while len(untrained_sample_indices):
        # Start training
        print("Starting Training")
        net.train()
        training_acc = []
        training_loss = []
        epoch = 0
        counter = 0
        while epoch < epochs:
            train_acc, train_loss = train_epoch(train_loader, gpu, optimizer, net, loss_metric, epoch)
            training_acc.append(train_acc)
            training_loss.append(train_loss)
            epoch += 1
            scheduler.step(train_loss)
            if len(training_loss) > 1 and train_loss < min(training_loss[:-1]):
                counter = 0
            else:
                counter += 1
            if counter > 2:
                print('Stopping Early')
                break

        decision_loader, done = get_decision_layer_loader(train_loader, batch_size, net, gpu, untrained_sample_indices,
                                                    training)

        if done:
            break

        net.train()
        minimal_trained_indices = train_decision_layer(net, decision_loader, gpu, loss_metric2, untrained_sample_indices, training, batch_size)
        untrained_sample_indices = np.setdiff1d(untrained_sample_indices, minimal_trained_indices)

        train_loader = get_train_loader(batch_size, untrained_sample_indices, training)

        if save_params:
            # torch.save(net, os.path.join('.', 'models', model_name + '_exit' + str(phase + 1)))
            torch.save(training_acc,
                       os.path.join('.', 'results', model_name + '_exit' + str(phase + 1) + '_learning_curve'))

        phase += 1

        # Create network
        net.add_phase()

        # Test if GPU is available
        gpu = torch.cuda.is_available()
        if gpu:
            print("CUDA available")
            net = net.cuda()

        # Create optimizer
        optimizer = torch.optim.Adam([{'params': net.features[-1].parameters()},
                                      {'params': net.classifier1.parameters()}
                                      ], lr=0.001, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)

        total_untrained_samples = len(untrained_sample_indices)

        print("Remaining Untrained Samples:", total_untrained_samples)
        training_samples.append(total_untrained_samples)

    torch.save(net, os.path.join('.', 'models', model_name + '_exit' + str(phase)))
    torch.save(training_samples, os.path.join('.', 'results', model_name + '_samples_trained'))


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch Size")
    parser.add_argument('--save_params', dest='save_params', action='store_true', default=False)
    parser.add_argument('--epochs', help='epochs for each phase', type=int, default=100)
    parser.add_argument('--method', type=str, default='layer_wise')
    parser.add_argument('--dataset', type=str, default='MNIST')




    args = parser.parse_args()

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)

    print()



