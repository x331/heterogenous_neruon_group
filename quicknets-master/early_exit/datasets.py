from torch.utils.data import Dataset
import torchvision.datasets as dataset


class IndexedMNIST(Dataset):
    def __init__(self, root, train, transform, download):
        self.data = dataset.MNIST(root=root, download=download, train=train, transform=transform)

    def __getitem__(self, index):
        data, target = self.data[index]
        return data, target, index

    def __len__(self):
        return len(self.data)


class IndexedFashionMNIST(Dataset):
    def __init__(self, root, train, transform, download):
        self.data = dataset.FashionMNIST(root=root, download=download, train=train, transform=transform)

    def __getitem__(self, index):
        data, target = self.data[index]
        return data, target, index

    def __len__(self):
        return len(self.data)


class IndexedCIFAR10(Dataset):
    def __init__(self, root, train, transform, download):
        self.data = dataset.CIFAR10(root=root, download=download, train=train, transform=transform)

    def __getitem__(self, index):
        data, target = self.data[index]
        return data, target, index

    def __len__(self):
        return len(self.data)


class IndexedCIFAR100(Dataset):
    def __init__(self, root, train, transform, download):
        self.data = dataset.CIFAR100(root=root, download=download, train=train, transform=transform)

    def __getitem__(self, index):
        data, target = self.data[index]
        return data, target, index

    def __len__(self):
        return len(self.data)


class IndexedImageNet(Dataset):
    def __init__(self, root, split, transform):
        self.data = dataset.ImageNet(root=root, split=split, transform=transform)

    def __getitem__(self, index):
        data, target = self.data[index]
        return data, target, index

    def __len__(self):
        return len(self.data)