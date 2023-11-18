from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].astype(np.int32).reshape(1400,)
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()

class Hdigit(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        scaler = MinMaxScaler()
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = scaler.fit_transform(data['data'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['data'][0][1].T.astype(np.float32))

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx] 
        x2 = self.V2[idx] 
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Prokar(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)

    def __len__(self):
        return 551

    def __getitem__(self, idx):

        x1 = self.V1[idx] 
        x2 = self.V2[idx]
        x3 = self.V3[idx] 
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "MNIST_USPS":
        dataset = MNIST_USPS('data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "Fashion":
        dataset = Fashion('data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Hdigit":
        dataset = Hdigit('data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "Prokaryotic":
        dataset = Prokar('data/')
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4

    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
