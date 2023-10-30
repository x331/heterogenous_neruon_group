import numpy as np
from tqdm import tqdm
import torch
from ptflops import get_model_complexity_info


def get_results(loader, thresholds, method, nets, break_ties='max_confidence', num_classes=10):
    correct = 0
    total = 0
    not_classified_exits = np.zeros(len(nets))
    sample_exits = np.zeros(len(nets))

    if method == 'decision':
        nets_o1 = []
        nets_o2 = []
        nets_conf = []
        for i, net in enumerate(nets):
            outputs1 = np.zeros(len(loader.dataset))
            outputs2 = np.zeros(len(loader.dataset))
            confidences = np.zeros(len(loader.dataset))
            for (x, y, indices) in tqdm(loader):
                x = x.cuda()
                o1, o2 = net.forward(x)
                o2 = o2[:, 0]
                confidence = 1 / num_classes * torch.sum(
                    torch.softmax(o1, dim=1) * torch.log(torch.softmax(o1, dim=1)), dim=1)
                pred = o1.data.max(1)[1]  # get the index of the max log-probability
                pred2 = torch.nn.functional.sigmoid(o2).cpu().detach().numpy()
                outputs1[indices] = pred.cpu()
                outputs2[indices] = pred2
                confidences[indices] = confidence.cpu().detach()

            nets_o1.append(outputs1)
            nets_o2.append(outputs2)
            nets_conf.append(confidences)

        nets_o1 = np.array(nets_o1)
        nets_o2 = np.array(nets_o2)
        nets_conf = np.array(nets_conf)

        exited = np.zeros(nets_o1.shape[1])

        ys = np.zeros(len(loader.dataset))

        for (x, y, indices) in loader:
            ys[indices] = y

        exit_accuracies = []
        for i, arr in enumerate(nets_conf):
            indices = np.where((arr > thresholds[i]) * (exited == 0) * (nets_o2[i] > 0.5))
            indices = indices[0]
            sample_exits[i] += len(indices)
            exited[indices] = 1
            correct_array = nets_o1[i][indices] == ys[indices]
            correct += correct_array.sum()
            total += len(indices)

            exit_accuracy = correct_array.sum() / len(indices)
            exit_accuracies.append(exit_accuracy)

        if break_ties == 'max_confidence':
            indices = np.where(exited == 0)
            indices = indices[0]
            exit_indices = np.argmax(nets_conf[:, indices], axis=0)

            correct_array = nets_o1[exit_indices, indices] == ys[indices]
            not_classified_exits = np.bincount(exit_indices)
            correct += correct_array.sum()
            total += len(indices)

        elif break_ties == 'max_votes':
            indices = np.where(exited == 0)
            indices = indices[0]
            arr = nets_o1[:, indices]
            axis = 0
            u, indices = np.unique(arr, return_inverse=True)
            pred = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(arr.shape),
                                                   None, np.max(indices) + 1), axis=axis)]
            indices = np.where(exited == 0)
            indices = indices[0]
            correct_array = pred == ys[indices]
            correct += correct_array.sum()
            total += len(indices)
        elif break_ties == 'end':
            indices = np.where(exited == 0)
            indices = indices[0]

            correct_array = nets_o1[-1][indices] == ys[indices]
            correct += correct_array.sum()
            total += len(indices)

        accuracy = correct / total

        return exit_accuracies, accuracy, sample_exits, not_classified_exits

    elif method == 'entropy':
        nets_o1 = []
        nets_o2 = []
        for i, net in enumerate(nets):
            outputs1 = np.zeros(len(loader.dataset))
            outputs2 = np.zeros(len(loader.dataset))
            for (x, y, indices) in tqdm(loader):
                x = x.cuda()
                o1, _ = net.forward(x)
                pred = o1.data.max(1)[1]  # get the index of the max log-probability
                entropy = 1 / num_classes * torch.sum(
                    torch.softmax(o1, dim=1) * torch.log(torch.softmax(o1, dim=1)), dim=1).cpu().detach().numpy()
                outputs1[indices] = pred.cpu()
                outputs2[indices] = entropy

            nets_o1.append(outputs1)
            nets_o2.append(outputs2)

        nets_o1 = np.array(nets_o1)
        nets_o2 = np.array(nets_o2)

        exited = np.zeros(nets_o1.shape[1])

        ys = np.zeros(len(loader.dataset))

        for (x, y, indices) in loader:
            ys[indices] = y

        exit_accuracies = []
        for i, arr in enumerate(nets_o2):
            indices = np.where((arr > thresholds[i]) * (exited == 0))
            indices = indices[0]
            sample_exits[i] += len(indices)
            exited[indices] = 1
            correct_array = nets_o1[i][indices] == ys[indices]
            correct += correct_array.sum()
            total += len(indices)

            exit_accuracy = correct_array.sum() / len(indices)
            exit_accuracies.append(exit_accuracy)

        if break_ties == 'max_confidence':
            indices = np.where(exited == 0)
            indices = indices[0]
            exit_indices = np.argmax(nets_o2[:, indices], axis=0)

            correct_array = nets_o1[exit_indices, indices] == ys[indices]
            not_classified_exits = np.bincount(exit_indices)
            correct += correct_array.sum()
            total += len(indices)

        elif break_ties == 'max_votes':
            indices = np.where(exited == 0)
            indices = indices[0]
            arr = nets_o1[:, indices]
            axis = 0
            u, indices = np.unique(arr, return_inverse=True)
            pred = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(arr.shape),
                                                   None, np.max(indices) + 1), axis=axis)]
            indices = np.where(exited == 0)
            indices = indices[0]
            correct_array = pred == ys[indices]
            correct += correct_array.sum()
            total += len(indices)

        elif break_ties == 'end':
            indices = np.where(exited == 0)
            indices = indices[0]

            correct_array = nets_o1[-1][indices] == ys[indices]
            correct += correct_array.sum()
            total += len(indices)

        accuracy = correct/total

        return exit_accuracies, accuracy, sample_exits, not_classified_exits

def get_macs(net):
    macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
    return macs, params

def get_training_flops(nets, epochs):

    #(add-multiplies per forward pass) * (2 FLOPs/add-multiply) * (3 for forward and backward pass) * (number of examples in dataset) * (number of epochs)

    macs = []
    flops = []
    for i, net in enumerate(nets):
        mac, _ = get_macs(net)
        macs.append(mac)
        if i == 0:
            f = mac * 2 * 3 * 50000 * epochs[i]
            flops.append(f)
        else:
            f = mac * 2 * 50000 * epochs[i]
            f += (mac-macs[-2]) * 2 * 2 * 50000 * epochs[i]
            flops.append(f)

    return flops


def get_testing_flops(nets, samples, not_classified_exit):

    #(add-multiplies per forward pass) * (2 FLOPs/add-multiply) * (3 for forward and backward pass) * (number of examples in dataset) * (number of epochs)

    macs = []
    flops = []
    for i, net in enumerate(nets):
        mac, _ = get_macs(net)
        macs.append(mac)
        f = mac * 2 * samples[i]
        flops.append(f)
        if i == len(nets) - 1:
            f = mac * 2 * sum(not_classified_exit)
            flops.append(f)


    return flops