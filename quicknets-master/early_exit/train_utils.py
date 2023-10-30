import torch
import numpy as np
import time
from torch.utils.data import SubsetRandomSampler
from functools import reduce

def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.nn.functional.sigmoid(output)
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output.clamp(min=1e-12))) + weights[0] * (
                    (1 - target) * torch.log((1 - output).clamp(min=1e-12)))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    if torch.isnan(torch.neg(torch.mean(loss))):
        print(weights[1] * (target * torch.log(output)), weights[0] * ((1 - target) * torch.log(1 - output)))
        print('==============')
        print((1 - target), torch.log(1 - output), weights[0])
        input(loss)
    return torch.neg(torch.mean(loss))


def train_decision_layer(net, decision_loader, gpu, loss_metric2, untrained_sample_indices, training_dataset, batch_size, num_classes, threshold):

    accuracy = 0
    net.train()
    optimizer = torch.optim.Adam(net.classifier2.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    training_loss = []
    counter = 0
    for epoch in range(100):
        correct1 = 0
        correct2 = 0
        train_loss = 0
        total = 0
        for i, (x, y, indices) in enumerate(decision_loader):
            if gpu:
                x = x.cuda()
                y = y.cuda()
            # loss calculation and gradient update:
            optimizer.zero_grad()
            outputs1, output2 = net.forward(x)
            output2 = output2[:, 0]
            pred = outputs1.data.max(1)[1]  # get the index of the max log-probability
            correct_array = pred.eq(y.data).cpu()
            label = correct_array.float().detach().cuda()
            loss2 = loss_metric2(output2, label)
            # loss2 = loss_metric2(output2, label)

            correct1 += correct_array.sum()
            loss = loss2
            loss.backward()
            optimizer.step()
            pred2 = torch.nn.functional.sigmoid(output2) > 0.5
            # pred2 = torch.nn.functional.sigmoid(output2) > 0.5

            pred2 = pred2.int()
            correct_array2 = pred2.eq(label.int().data).cpu()
            correct2 += correct_array2.sum()
            total += x.shape[0]


            train_loss += loss
            print(list(net.classifier2.parameters()))
        training_loss.append(train_loss)
        print("Classifier network epoch", epoch, "Accuarcy: ", correct1 / total)
        print("Cluster network epoch", epoch, "Accuarcy: ", correct2 / total)
        print("Loss:", train_loss)
        scheduler.step(train_loss)

        if len(training_loss) > 1 and train_loss < min(training_loss[:-1]):
            counter = 0
        else:
            counter += 1
        if counter > 2:
            print('Stopping Early')
            break

    # Calculating the accuracy of classifier 2

    net.eval()
    trained_indices = []
    train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(untrained_sample_indices))
    for (x, y, indices) in train_loader:

        if gpu:
            x = x.cuda()
            y = y.cuda()

        # loss calculation and gradient update:
        outputs1, outputs2 = net.forward(x)
        outputs2 = outputs2[:, 0]
        pred = outputs1.data.max(1)[1]  # get the index of the max log-probability
        confidence = 1 / 10 * torch.sum(torch.softmax(outputs1, dim=1) * torch.log(torch.softmax(outputs1, dim=1)), dim=1)
        pred2 = torch.nn.functional.sigmoid(outputs2)
        correct_array = pred.eq(y.data)
        indices_to_exit = reduce(np.intersect1d, ((pred2 > 0.8).detach().cpu().nonzero(as_tuple=True)[0],
                                                  correct_array.detach().cpu().nonzero(as_tuple=True)[0],
                                                  (confidence > threshold).detach().cpu().nonzero(as_tuple=True)[0]))
        trained_indices += indices[indices_to_exit].tolist()

    print("Trained Samples: ", len(trained_indices), '/', len(untrained_sample_indices))
    trained_indices = np.array(trained_indices).astype(int)
    print('Trained indices:', trained_indices.shape)
    net.train()

    print('---------------------------------------------------------------------------------------------------')
    return trained_indices, threshold


def get_minimal_trained_indices(trained_indices, training_dataset):
    labels = torch.tensor(training_dataset.data.targets)
    trained_class_indices = []
    min_samples = int(len(trained_indices) / len(torch.unique(labels)))
    for i in range(1000):
        class_indices = (labels == i).int().nonzero(as_tuple=True)[0]
        trained_class_indices.append(np.intersect1d(class_indices, trained_indices))
        min_samples = min(len(trained_class_indices[-1]), min_samples)

    minimal_trained_indices = []
    for indices in trained_class_indices:
        minimal_trained_indices = np.concatenate((minimal_trained_indices, indices[:min_samples]), axis=0)
    return minimal_trained_indices


def train_epoch(train_loader, gpu, optimizer, net, loss_metric, epoch):
    time1 = time.time()  # timekeeping
    train_loss = 0
    correct = 0
    total = 0
    for i, (x, y, indices) in enumerate(train_loader):

        if gpu:
            x = x.cuda()
            y = y.cuda()
        # loss calculation and gradient update:
        optimizer.zero_grad()
        outputs1, _ = net.forward(x)
        loss = loss_metric(outputs1, y)
        pred = outputs1.data.max(1)[1]  # get the index of the max log-probability
        correct_array = pred.eq(y.data).cpu()
        correct += correct_array.sum()
        loss.backward()
        train_loss += loss
        optimizer.step()
        total += len(correct_array)

    print("Epoch", epoch, ':')
    print("Loss:", train_loss)
    accuracy = 100. * correct / total
    print("Accuracy:", accuracy)
    training_acc = accuracy
    time2 = time.time()  # timekeeping
    print('Elapsed time for epoch:', time2 - time1, 's')
    print('-------------------------------------------------------------------------------------------------------')
    return training_acc, train_loss


def train_epoch_joint_loss(train_loader, test_loader, gpu, optimizer, net, loss_metric, loss_metric2, epoch, loss_scale,
                           untrained_sample_indices, last_epoch, threshold_loss):
    time1 = time.time()  # timekeeping
    train_loss = 0
    correct = 0
    total = 0
    correct2 = 0
    for i, (x, y, indices) in enumerate(train_loader):

        if gpu:
            x = x.cuda()
            y = y.cuda()

        # loss calculation and gradient update:
        optimizer.zero_grad()
        outputs1, output2 = net.forward(x)
        output2 = output2[:, 0]
        loss = loss_metric(outputs1, y)
        if threshold_loss:
            loss = torch.nn.functional.threshold(loss,  0.015, 0)
            loss = torch.mean(loss)
        pred = outputs1.data.max(1)[1]  # get the index of the max log-probability
        correct_array = pred.eq(y.data).cpu()
        correct += correct_array.sum()
        label = correct_array.float().detach().cuda()
        # loss2 = loss_metric2(output2[torch.arange(output2.shape[0]), pred], label)
        loss2 = loss_metric2(output2, label)
        pred2 = torch.nn.functional.sigmoid(output2) > 0.5
        pred2 = pred2.int()
        correct_array2 = pred2.eq(label.int().data).cpu()
        correct2 += correct_array2.sum()
        total_loss = loss * loss_scale + (1 - loss_scale) * loss2
        total_loss.backward()
        train_loss += total_loss
        optimizer.step()
        total += len(correct_array)

    print("Epoch", epoch, ':')
    print("Loss:", train_loss)
    accuracy = 100. * correct / total
    print("Accuracy:", accuracy)
    print("Commitment layer accuracy:", 100. * correct2 / total)
    training_acc = accuracy
    time2 = time.time()  # timekeeping
    print('Elapsed time for epoch:', time2 - time1, 's')
    print('-------------------------------------------------------------------------------------------------------')

    net.eval()
    total = 0
    correct = 0
    for i, (x, y) in enumerate(test_loader):
        if gpu:
            x = x.cuda()
            y = y.cuda()

        outputs1, output2 = net.forward(x)
        pred = outputs1.data.max(1)[1]  # get the index of the max log-probability
        correct_array = pred.eq(y.data).cpu()
        correct += correct_array.sum()
        total += len(correct_array)

    testing_acc = 100. * correct / total

    minimal_trained_indices = []
    if last_epoch:
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
            # pred2 = torch.nn.functional.sigmoid(outputs2[torch.arange(outputs2.shape[0]), pred])
            pred2 = torch.nn.functional.sigmoid(outputs2[:, 0])
            indices_to_exit = (pred2 > 0.8).nonzero(as_tuple=True)[0]
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
        net.train()

        print('---------------------------------------------------------------------------------------------------')

    return training_acc, testing_acc, minimal_trained_indices
