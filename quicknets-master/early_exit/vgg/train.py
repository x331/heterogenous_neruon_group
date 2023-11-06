import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler
import os
import random
import argparse
import numpy as np
import sys
import time

sys.path.append('..')
print('hello')
from datasets import IndexedCIFAR10, IndexedCIFAR100
from train_utils import train_decision_layer, train_epoch, train_epoch_joint_loss, get_minimal_trained_indices
from torchvision import datasets, transforms

sys.path.append('../../')
from architectures.VGG_CIFAR100 import VGGEarlyExit, cfg


def get_trained_indices(net, train_loader, gpu, untrained_sample_indices):
    correct = 0
    total = 0
    net.eval()
    trained_indices = []
    for i, (x, y, indices) in enumerate(train_loader):

        if gpu:
            x = x.cuda()
            y = y.cuda()

        # loss calculation and gradient update:
        outputs1, outputs2 = net.forward(x)
        pred = outputs1.data.max(1)[1]  # get the index of the max log-probability
        entropy = torch.sum(
            torch.softmax(outputs1, dim=1) * torch.log(torch.softmax(outputs1, dim=1)), dim=1)

        indices_to_exit = (entropy > -0.1).nonzero(as_tuple=True)[0]
        trained_indices += indices[indices_to_exit].tolist()

        pred = pred[indices_to_exit]
        y_data = y[indices_to_exit]
        correct_array = pred.eq(y_data)
        correct += torch.sum(correct_array)
        total += len(indices_to_exit)
    print("Accuracy on Trained samples : ", correct / total)
    print("Trained Samples: ", total, '/', len(untrained_sample_indices))
    accuracy = correct / total
    minimal_trained_indices = get_minimal_trained_indices(trained_indices, train_loader.dataset)
    minimal_trained_indices = np.array(minimal_trained_indices).astype(int)
    print('Trained indices:', minimal_trained_indices.shape)
    print(
        '---------------------------------------------------------------------------------------------------')
    net.train()
    return minimal_trained_indices


def train(seed=0, batch_size=128, save_params=False, method='layer_wise', config='D', dataset='CIFAR10', lr=0.01,
          remove_data=False, lr_scheduler='cyclic', sample_learned_data=False, decision_layer=False,
          batch_norm=False, output_layers=[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1], weight_decay=5e-4):
    params = [dataset, seed, batch_size, method, config,  lr, remove_data, lr_scheduler, sample_learned_data,
              decision_layer, batch_norm, output_layers, weight_decay]
    model_name = '_'.join([str(x) for x in params])
    if output_layers == None:
        output_layers =[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]


    # Create directories
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists('models'):
        os.makedirs('models')

    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Create dataloaders
    if dataset == 'CIFAR100':
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

        testing = datasets.CIFAR100(root='../../data',
                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                                                              std=std)]),
                                    train=False,
                                    download=True)
        test_loader = torch.utils.data.DataLoader(dataset=testing, batch_size=batch_size, shuffle=True)
        # Create network
        net = VGGEarlyExit(batch_norm=False, config=cfg[config], num_class=100,
                           compressed_classifier=compressed_classifier)

    elif dataset == 'CIFAR10':
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

        testing = datasets.CIFAR10(root='../../data',
                                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                                                             std=std)]),
                                   train=False,
                                   download=True)
        test_loader = torch.utils.data.DataLoader(dataset=testing, batch_size=batch_size, shuffle=True)

        # Create network
        net = VGGEarlyExit(batch_norm=batch_norm, config=cfg[config], output_layers=output_layers)

    # Test if GPU is available
    gpu = torch.cuda.is_available()
    if gpu:
        print("CUDA available")

    loss_metric = nn.CrossEntropyLoss()
    loss_metric2 = nn.BCEWithLogitsLoss()

    if method == 'end_to_end' or method == 'backbone_first':
        net.set_phase(len(net.classifiers1) - 1)
        optimizer = torch.optim.Adam([{'params': net.frozen_features.parameters()},
                                      {'params': net.features.parameters()},
                                      {'params': net.classifier1.parameters()}
                                      ], lr=lr, weight_decay=weight_decay)
        if lr_scheduler == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr * 0.8, max_lr=lr * 1.2,
                                                          cycle_momentum=False)
        net = net.cuda()
        training_acc = []
        testing_acc = []
        epoch = 0
        while epoch < 50:
            train_acc, test_acc = train_epoch(train_loader, test_loader, gpu, optimizer, net, loss_metric, epoch)
            training_acc.append(train_acc)
            testing_acc.append(test_acc)
            epoch += 1

        if method == 'backbone_first':
            while net.phase > 0:
                net.set_phase(net.phase - 1)
                if gpu:
                    net = net.cuda()
                print("Training for phase ", net.phase)
                optimizer = torch.optim.Adam([{'params': net.classifier1.parameters()}
                                              ], lr=lr, weight_decay=weight_decay)
                epoch = 0
                while epoch < 50:
                    train_acc, test_acc = train_epoch(train_loader, test_loader, gpu, optimizer, net, loss_metric,
                                                      epoch)
                    training_acc.append(train_acc)
                    testing_acc.append(test_acc)
                    epoch += 1
                _ = train_decision_layer(net, train_loader, gpu, loss_metric2,
                                         untrained_sample_indices=torch.arange(len(train_loader.dataset)),
                                         balance_decision_layer=balance_decision_layer, train_acc=train_acc)

        if save_params:
            torch.save(net, os.path.join('.', 'models', model_name))
            torch.save(training_acc,
                       os.path.join('.', 'results', model_name + '_learning_curve'))
            torch.save(testing_acc,
                       os.path.join('.', 'results', model_name + '_testing_curve'))

        return

    net.cuda()
    # Create optimizer
    optimizer = torch.optim.Adam([
        {'params': net.features[-1].parameters()},
        {'params': net.classifier.parameters()}
    ], lr=lr, weight_decay=5e-4)

    if lr_scheduler == 'cyclic':
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr * 0.8, max_lr=lr * 1.2, cycle_momentum=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)

    untrained_sample_indices = torch.arange(len(train_loader.dataset))
    training_samples = [len(untrained_sample_indices)]
    epochs = []
    total_time = 0
    for phase in range(sum(output_layers)):

        # Start training
        print("Starting Training")
        net.train()
        training_acc = []
        training_loss = []
        epoch = 0
        start_time = time.time()
        counter = 0
        while epoch < 100:

            train_acc, loss = train_epoch(train_loader, gpu, optimizer, net, loss_metric, epoch)
            training_acc.append(float(train_acc))
            training_loss.append(float(loss))
            epoch += 1
            if lr_scheduler == 'cyclic':
                scheduler.step(loss)

            # if len(training_loss) > 7:
            #     if np.mean(training_loss[-8:-1]) < loss:
            #         print('Stopping Early')
            #
            #         if not decision_layer and remove_data:
            #             minimal_trained_indices = get_trained_indices(net, train_loader, gpu, untrained_sample_indices)
            #         break

            if len(training_loss) > 1 and loss < min(training_loss[:-1]):
                counter = 0
            else:
                counter += 1

            if counter > 3:
                print('Stopping Early')
                if not decision_layer and remove_data:
                    minimal_trained_indices = get_trained_indices(net, train_loader, gpu, untrained_sample_indices)
                break

        if epoch == 100 and not decision_layer and remove_data:
            minimal_trained_indices = get_trained_indices(net, train_loader, gpu, untrained_sample_indices)

        epochs.append(epoch)

        end_time = time.time()
        total_time += end_time - start_time

        if decision_layer:
            train_loader = torch.utils.data.DataLoader(dataset=training, batch_size=batch_size, shuffle=True)
            net.eval()
            total = 0
            correct = 0
            correct_indices = []
            for i, (x, y, indices) in enumerate(train_loader):
                if gpu:
                    x = x.cuda()
                    y = y.cuda()

                outputs1, output2 = net.forward(x)
                pred = outputs1.data.max(1)[1]  # get the index of the max log-probability
                correct_array = pred.eq(y.data).cpu()
                correct_indices_batch = correct_array.nonzero(as_tuple=True)[0]
                correct_indices += indices[correct_indices_batch].tolist()
                correct += correct_array.sum()
                total += len(correct_array)
            train_acc = 100. * correct / total
            net.train()
            weights = torch.ones(50000) * (train_acc / 100)
            weights[untrained_sample_indices] = 1 - (train_acc / 100)
            train_loader = torch.utils.data.DataLoader(dataset=training, batch_size=batch_size,
                                                       sampler=WeightedRandomSampler(weights, len(weights)))

            minimal_trained_indices = train_decision_layer(net, train_loader, gpu, loss_metric2,
                                                           untrained_sample_indices,
                                                           balance_decision_layer, train_acc)

        if remove_data:
            untrained_sample_indices = np.setdiff1d(untrained_sample_indices, minimal_trained_indices)
        # weights = torch.ones(50000) * 0.1
        # weights[untrained_sample_indices] = 1
        # train_loader = torch.utils.data.DataLoader(dataset=training, batch_size=batch_size,
        #                                            sampler=WeightedRandomSampler(weights, len(weights)))
        train_loader = torch.utils.data.DataLoader(dataset=training, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(untrained_sample_indices))

        if save_params:
            torch.save(training_acc,
                       os.path.join('.', 'results', model_name + '_exit' + str(net.phase) + '_learning_curve'))

        net.add_phase()
        if gpu:
            net = net.cuda()

        # Create optimizer
        optimizer = torch.optim.Adam([
            {'params': net.features[-1].parameters()},
            {'params': net.classifier.parameters()}
        ], lr=lr, weight_decay=5e-4)

        if lr_scheduler == 'cyclic':
            # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr * 0.8, max_lr=lr * 1.2, cycle_momentum=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)

        total_untrained_samples = len(untrained_sample_indices)

        print("Remaining Untrained Samples:", total_untrained_samples)
        training_samples.append(total_untrained_samples)

        print('Elapsed time for phase:', total_time / 60.0, 'mins')

        if save_params:
            torch.save(net, os.path.join('.', 'models', model_name + '_exit' + str(phase)))
            torch.save(training_samples, os.path.join('.', 'results', model_name + '_samples_trained'))
            torch.save(epochs, os.path.join('.', 'results', model_name + 'epochs_trained'))

            if save_params:
                torch.save(total_time/60.0, os.path.join('.', 'results', model_name + '_time'))


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
    parser.add_argument('--save_params', dest='save_params', action='store_true', default=False)
    parser.add_argument('--method', type=str, default='layer_wise')
    parser.add_argument('--config', type=str, default='D', help='VGG config')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR 10 or CIFAR100')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--remove_data', dest='remove_data', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', type=str, default='cyclic')
    parser.add_argument('--decision_layer', dest='decision_layer', action='store_true', default=False)
    parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', default=False)
    parser.add_argument('--output_layers', type=int, nargs='+', help='exits in the network')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight Decay')


    args = parser.parse_args()

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)

    print()
