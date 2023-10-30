import torch
import argparse
import os
from torchvision import datasets, transforms
import sys
import numpy as np
sys.path.append('../../')
from architectures.VGG_CIFAR100 import VGGEarlyExit, cfg
sys.path.append('../')
from plotting_utils import get_learning_curve_graph, get_figures
from datasets import IndexedCIFAR10
from eval_utils import get_training_flops, get_results, get_testing_flops

# [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M']

def eval(seed=0, batch_size=128, config='A', dataset='CIFAR10', method='decision', train_method='layer_wise', lr=0.01,
         remove_data=False, lr_scheduler='cyclic', sample_learned_data=False, decision_layer=False, batch_norm=False,
         output_layers=[0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1], weight_decay=5e-4):
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    training = IndexedCIFAR10(root='../../data',
                              transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                                                        std=std)]),
                              train=True,
                              download=True)
    train_loader = torch.utils.data.DataLoader(dataset=training, batch_size=64, shuffle=False)
    testing = IndexedCIFAR10(root='../../data',
                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                                                       std=std)]),
                             train=False,
                             download=True)
    test_loader = torch.utils.data.DataLoader(dataset=testing, batch_size=64, shuffle=False)

    nets = []
    params = [dataset, seed, batch_size, train_method, config, lr, remove_data, lr_scheduler, sample_learned_data,
              decision_layer, batch_norm, output_layers, weight_decay]
    model_name = '_'.join([str(x) for x in params])
    exits = sum(output_layers)
    for i in range(exits):
        network = torch.load('models/'+model_name+'_exit'+str(exits-1))
        network.set_phase(i+1)
        network.eval()
        network.cuda()
        nets.append(network)

    epochs = torch.load('results/'+model_name+'epochs_trained')

    # training_flops = get_training_flops(nets, epochs)
    # print('Training Flops:', sum(training_flops)/10**15, 'P-FLOPS')
    #
    # exit_accuracy, accuracy, sample_exits, not_classified_exits = get_results(train_loader, np.ones(exits)*-0.1,
    #                                                                           'entropy', nets)
    # print('Exit Accuracy:', exit_accuracy)
    # print('Training Accuracy:', accuracy)
    # print('Sample Exits:', sample_exits)
    # print('Not classified Exits:', not_classified_exits)

    # exit_accuracy, accuracy, sample_exits, not_classified_exits = get_results(test_loader, np.ones(exits) * -0.2,
    #                                                                           'entropy', nets)
    # print('Exit Accuracy:', exit_accuracy)
    # print('Testing Accuracy:', accuracy)
    # print('Sample Exits:', sample_exits)
    # print('Not classified Exits:', not_classified_exits)
    #
    # testing_flops = get_testing_flops(nets, sample_exits, not_classified_exits)
    # print('Testing Flops:', sum(testing_flops) / 10 ** 15, 'P-FLOPS')

    total = 0
    correct = 0
    for (x, y, indices) in test_loader:
        x = x.cuda()
        y = y
        o1 = nets[-1].forward(x)
        pred = o1.data.max(1)[1]  # get the index of the max log-probability
        c = pred.cpu() == y
        correct += c.sum()
        total += len(y)
    print("Accuracy:", correct/total)


    # results = {
    #     'fig1': fig1,
    #     'fig2': fig2,
    #     'fig3': fig3,
    #     'train_accuracy': train_accuracy,
    #     'test_accuracy': test_accuracy,
    #     'exit_train_accuracy': exit_train_accuracy,
    #     'exit_test_accuracy': exit_test_accuracy,
    #     'exits': exits,
    #     'test_exits': test_exits,
    #     'not_classified_exits': not_classified_exits,
    #     'test_not_classified_exits':test_not_classified_exits
    # }
    #
    # torch.save(results, os.path.join('.', 'results', model_name +'_' + str(threshold)+ '_' + method +'_eval'))


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
    parser.add_argument('--train_method', type=str, default='layer_wise')
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

    eval(**args)

    print()
