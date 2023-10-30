import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import time
import pandas as pd
import argparse
import sys
import numpy as np
sys.path.append('../')
from datasets import IndexedCIFAR10
from torchvision import datasets, transforms
sys.path.append('../../')
from architectures.VGG_CIFAR100 import vgg11


def train(seed=0, batch_size=64, save_params=False, lr=0.0003, lr_scheduler='cyclic', measure_time=False):

    network = 'CIFAR10_VGG11_standard'
    params = [network, seed, batch_size, lr, lr_scheduler, measure_time]
    model_name = '_'.join([str(x) for x in params])

    # Create directories
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists('models'):
        os.makedirs('models')

    # Set seed
    torch.manual_seed(seed)

    # Create dataloaders
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    training = datasets.CIFAR10(root='../../data',
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
    net = vgg11(10)

    # Test if GPU is available
    gpu = torch.cuda.is_available()
    if gpu:
        print("CUDA available")
        net = net.cuda()

    # Create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr * 0.8, max_lr=lr * 1.2,
                                                  cycle_momentum=False)
    loss_metric = nn.CrossEntropyLoss()

    # Start training
    print("Starting Training")

    if measure_time:
        start_time = time.time()
    net.train()
    training_acc = []
    testing_acc = []
    epochs = 50
    for epoch in range(epochs):
        time1 = time.time()  # timekeeping
        train_loss = 0
        correct = 0

        for i, (x, y) in enumerate(train_loader):

            if gpu:
                x = x.cuda()
                y = y.cuda()

            # loss calculation and gradient update:
            optimizer.zero_grad()
            outputs = net.forward(x)
            loss = loss_metric(outputs, y)
            pred = outputs.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data).cpu().sum()
            loss.backward()
            optimizer.step()
            train_loss += loss

        print("Epoch", epoch + 1, ':')
        print("Loss:", train_loss)
        accuracy = 100. * correct / len(train_loader.dataset)
        print("Accuracy:", accuracy)
        training_acc.append(accuracy)

        time2 = time.time()  # timekeeping
        print('Elapsed time for epoch:', time2 - time1, 's')
        print('ETA of completion:', (time2 - time1) * (epochs - epoch - 1) / 60, 'minutes')
        print('-------------------------------------------------------------------------------------------------------')
        # Calculate test accuracy
        training_accuracy = accuracy

        if not measure_time:
            net.eval()
            correct = 0
            for i, (x, y) in enumerate(test_loader):

                if gpu:
                    x = x.cuda()
                    y = y.cuda()

                # loss calculation and gradient update:
                outputs = net.forward(x)
                pred = outputs.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(y.data).cpu().sum()

            net.train()
            testing_acc.append(correct/len(test_loader.dataset))

            time2 = time.time()  # timekeeping
            print('Elapsed time for epoch:', time2 - time1, 's')
            print('ETA of completion:', (time2 - time1) * (epochs - epoch - 1) / 60, 'minutes')
            print('-------------------------------------------------------------------------------------------------------')

        if len(training_acc) > 3:
            if np.mean(training_acc[-4:-1]) > training_accuracy:
                print('Stopping Early')
                break

        if lr_scheduler == 'cyclic':
            scheduler.step()

    if measure_time:
        end_time = time.time()
        print('Elapsed time for epoch:', (end_time - start_time)/60.0, 'mins')
        if save_params:
            torch.save((end_time - start_time)/60.0, os.path.join('.', 'results', model_name + '_time'))

    if save_params:
        torch.save(net.state_dict(), os.path.join('.', 'models', model_name + '_' + str(epoch)))
        torch.save(training_acc, os.path.join('.', 'results', model_name + '_learning_curve'))
        torch.save(testing_acc, os.path.join('.', 'results', model_name + '_testing_curve'))


    # net.eval()
    # correct = 0
    # for i, (x, y) in enumerate(test_loader):
    #     if gpu:
    #         x = x.cuda()
    #         y = y.cuda()
    #     outputs = net.forward(x)
    #     pred = outputs.data.max(1)[1]
    #     correct += pred.eq(y.data).cpu().sum()
    #
    # accuracy = 100. * float(correct) / len(test_loader.dataset)
    #
    # df = pd.DataFrame({'epochs': [epochs],
    #                    'seed': [seed],
    #                    'batch_size': [batch_size],
    #                    'training_accuracy': [training_accuracy],
    #                    'accuracy': [accuracy],
    #                    })
    #
    # df.to_csv(os.path.join('.', 'results', network + '_test_accuracy.csv'), mode='a',
    #           header=not os.path.exists(os.path.join('.', 'results', network + '_test_accuracy.csv')))



if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch Size")
    parser.add_argument('--save_params',  dest='save_params', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0003, help="Learning Rate")
    parser.add_argument('--lr_scheduler', type=str, default='cyclic')
    parser.add_argument('--measure_time',  dest='measure_time', action='store_true', default=False)



    args = parser.parse_args()

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)

    print()