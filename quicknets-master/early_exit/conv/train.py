import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import sys
import random
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler
import pandas as pd
sys.path.append('../')
from datasets import IndexedCIFAR10, IndexedCIFAR100
from train_utils import train_decision_layer, train_epoch
from torchvision import datasets, transforms
sys.path.append('../../')
from architectures.conv import AdaptiveConv


def get_decision_layer_loader(batch_size, net, gpu, untrained_sample_indices, dataset, num_classes):
    df = pd.DataFrame(columns=['confidence', 'correct', 'index'])
    labels = torch.tensor(dataset.data.targets)
    net.eval()
    correct = 0
    total = 0
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(untrained_sample_indices))

    for i, (x, y, indices) in enumerate(train_loader):

        if gpu:
            x = x.cuda()
            y = y.cuda()

        # loss calculation and gradient update:
        outputs1, outputs2 = net.forward(x)
        confidence = 1 / num_classes * torch.sum(torch.softmax(outputs1, dim=1) * torch.log(torch.softmax(outputs1, dim=1)), dim=1)
        pred = outputs1.data.max(1)[1]  # get the index of the max log-probability
        correct_array = pred.eq(y)
        correct += torch.sum(correct_array)
        total += len(y)
        for i, c in enumerate(correct_array):
            df.loc[len(df.index)] = [confidence[i].cpu().detach(), correct_array[i].cpu().detach(), int(indices[i])]

    net.train()

    if correct - total == 0:
        return None, True, -100

    threshold_df = pd.DataFrame(columns=['threshold', 'accuracy', 'samples'])
    for t in range(100):
        threshold = t * 0.002 - 0.19
        above_threshold = df.loc[df['confidence'] > threshold]
        if len(above_threshold):
            accuracy = sum(above_threshold['correct'] == True) / len(above_threshold)
            threshold_df.loc[len(threshold_df.index)] = [threshold, accuracy, len(above_threshold)]

    threshold = 0
    for idx, row in threshold_df.iterrows():
        if row['accuracy'] > 0.9:
            threshold = row['threshold']
            break
    print('------------------------------------------------')
    print('threshold:', threshold)
    index_weights = torch.zeros(len(dataset))
    above_threshold = df.loc[df['confidence'] > threshold]
    correct = sum(above_threshold['correct'] == True)
    index_weights[above_threshold.loc[above_threshold['correct'] == True]['index'].tolist()] = 1 / correct
    index_weights[above_threshold.loc[above_threshold['correct'] == False]['index'].tolist()] = 1 / (
                len(above_threshold) - correct)
    decision_layer_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                                        sampler=WeightedRandomSampler(
                                                            index_weights.type('torch.DoubleTensor'), len(dataset),
                                                            replacement=True))

    print('weight for untrained samples:',  1 / correct)
    print('weight for trained samples:', 1 / (len(above_threshold) - correct))
    print('------------------------------------------------')

    # loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
    #                                      sampler=SubsetRandomSampler(np.setdiff1d(untrained_sample_indices, trained_indices)))

    return decision_layer_loader, False, threshold


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


def train(seed=0, batch_size=128, save_params=False, epochs=100, dataset='CIFAR10', block=1, decision_layer=False):

    params = [dataset, seed, batch_size, epochs, block, decision_layer]
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
    if dataset == 'CIFAR10':
        num_classes = 10
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

    elif dataset == 'CIFAR100':
        num_classes = 100
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        training = IndexedCIFAR100(root='../../data',
                                  transform=transforms.Compose([transforms.RandomCrop(size=(32, 32), padding=4),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=mean,
                                                                                     std=std)]),
                                  train=True,
                                  download=True)
        train_loader = torch.utils.data.DataLoader(dataset=training, batch_size=batch_size, shuffle=True)

    # Create network
    net = AdaptiveConv(64, num_class=num_classes)

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
    decision_threshold = []
    while len(untrained_sample_indices):
        # Start training
        print("Starting Training")
        net.train()
        training_acc = []
        training_loss = []
        epoch = 0
        counter = 0
        while epoch < 2:
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

        # decision layer training
        if decision_layer:
            decision_loaders, done, threshold = get_decision_layer_loader(batch_size, net, gpu, untrained_sample_indices, training,
                                                               num_classes)

            if done:  # if all samples are correctly classified
                break

            net.train()
            trained_indices, threshold = train_decision_layer(net, decision_loaders, gpu, loss_metric2, untrained_sample_indices, training, batch_size, num_classes, threshold)

            decision_threshold.append(threshold)
            untrained_sample_indices = np.setdiff1d(untrained_sample_indices, trained_indices)

        if save_params:
            torch.save(training_acc,
                       os.path.join('.', 'results', model_name + '_exit' + str(phase) + '_learning_curve'))
            torch.save(training_loss,
                       os.path.join('.', 'results', model_name + '_exit' + str(phase) + '_loss'))
            torch.save(net, os.path.join('.', 'models', model_name + '_exit' + str(phase)))

        phase += 1
        if phase > block:
            break
            # Create network
        net.add_phase()
        #     net.add_block()

        train_loader = get_train_loader(batch_size, untrained_sample_indices, training, num_classes)

        # Test if GPU is available
        gpu = torch.cuda.is_available()
        if gpu:
            print("CUDA available")
            net = net.cuda()

        # Create optimizer
        optimizer = torch.optim.Adam([{'params': net.classifier1.parameters()}
                                      ], lr=0.001, weight_decay=5e-4)

        for features in net.features[-net.sub_phase - 1: ]:
            optimizer.add_param_group({'params': features.parameters()})

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)

        total_untrained_samples = len(untrained_sample_indices)

        print("Remaining Untrained Samples:", total_untrained_samples)
        training_samples.append(total_untrained_samples)

    torch.save(net, os.path.join('.', 'models', model_name + '_exit' + str(phase - 1)))
    torch.save(training_samples, os.path.join('.', 'results', model_name + '_samples_trained'))
    torch.save(decision_threshold, os.path.join('.', 'results', model_name + '_decision_thresholds'))


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
    parser.add_argument('--save_params', dest='save_params', action='store_true', default=False)
    parser.add_argument('--epochs', help='max epochs for each phase', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--block', type=int, default=1, help="max blocks to train")
    parser.add_argument('--decision_layer', dest='decision_layer', action='store_true', default=False,
                        help="train decision layer")


# checklist for adding new param:
    # 1. Add to parser
    # 2. Add to function definition
    # 3. Add to model name




    args = parser.parse_args()

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)

    print()



