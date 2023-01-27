import os
import logging
# from turtle import down
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
# import random
# from sklearn.metrics import confusion_matrix
# from torch.utils.data import DataLoader

# from model import *
from .niid_datasets import  FEMNIST
# from math import sqrt

# import torch.nn as nn

# import torch.optim as optim
# import torchvision.utils as vutils
# import time
import random

# from models.mnist_model import Generator, Discriminator, DHead, QHead
# from config import params
# import sklearn.datasets as sk
# from sklearn.datasets import load_svmlight_file

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

# def load_mnist_data(datadir, transform = None):

#     # transform = transforms.Compose([transforms.ToTensor()])

#     mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
#     mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

#     X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
#     X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

#     X_train = X_train.data.numpy()
#     y_train = y_train.data.numpy()
#     X_test = X_test.data.numpy()
#     y_test = y_test.data.numpy()

#     return (X_train, y_train, X_test, y_test)

# def load_fmnist_data(datadir):

#     transform = transforms.Compose([transforms.ToTensor()])

#     mnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
#     mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

#     X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
#     X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

#     X_train = X_train.data.numpy()
#     y_train = y_train.data.numpy()
#     X_test = X_test.data.numpy()
#     y_test = y_test.data.numpy()

#     return (X_train, y_train, X_test, y_test)

# def load_svhn_data(datadir):

#     transform =  None #transforms.Compose([transforms.ToTensor()])

#     svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
#     svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

#     X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
#     X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

#     # X_train = X_train.data.numpy()
#     # y_train = y_train.data.numpy()
#     # X_test = X_test.data.numpy()
#     # y_test = y_test.data.numpy()

#     return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir,transform =  None):

      #transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_celeba_data(datadir, transform =  None):

    # transform = None  #transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train =  celeba_train_ds.attr[:,gender_index:gender_index+1].reshape(-1)
    y_test = celeba_test_ds.attr[:,gender_index:gender_index+1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)

def load_femnist_data(datadir, download=True, transform=None):
    # transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, target_transform=transform, download=download)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, target_transform=transform,  download=download)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_clients, beta=0.4, download=True, transform=None):
    #np.random.seed(2020)
    #torch.manual_seed(2020)

    # if dataset == 'mnist':
    #     X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    # elif dataset == 'fmnist':
    #     X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir, transform=transform)
    # elif dataset == 'svhn':
    #     X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir, transform=transform)
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir, download, transform=transform)
    # elif dataset == 'generated':
    #     X_train, y_train = [], []
    #     for loc in range(4):
    #         for i in range(1000):
    #             p1 = random.random()
    #             p2 = random.random()
    #             p3 = random.random()
    #             if loc > 1:
    #                 p2 = -p2
    #             if loc % 2 ==1:
    #                 p3 = -p3
    #             if i % 2 == 0:
    #                 X_train.append([p1, p2, p3])
    #                 y_train.append(0)
    #             else:
    #                 X_train.append([-p1, -p2, -p3])
    #                 y_train.append(1)
    #     X_test, y_test = [], []
    #     for i in range(1000):
    #         p1 = random.random() * 2 - 1
    #         p2 = random.random() * 2 - 1
    #         p3 = random.random() * 2 - 1
    #         X_test.append([p1, p2, p3])
    #         if p1>0:
    #             y_test.append(0)
    #         else:
    #             y_test.append(1)
    #     X_train = np.array(X_train, dtype=np.float32)
    #     X_test = np.array(X_test, dtype=np.float32)
    #     y_train = np.array(y_train, dtype=np.int32)
    #     y_test = np.array(y_test, dtype=np.int64)
    #     idxs = np.linspace(0,3999,4000,dtype=np.int64)
    #     batch_idxs = np.array_split(idxs, n_clients)
    #     net_dataidx_map = {i: batch_idxs[i] for i in range(n_clients)}
    #     mkdirs("data/generated/")
    #     np.save("data/generated/X_train.npy",X_train)
    #     np.save("data/generated/X_test.npy",X_test)
    #     np.save("data/generated/y_train.npy",y_train)
    #     np.save("data/generated/y_test.npy",y_test)
    
    #elif dataset == 'covtype':
    #    cov_type = sk.fetch_covtype('./data')
    #    num_train = int(581012 * 0.75)
    #    idxs = np.random.permutation(581012)
    #    X_train = np.array(cov_type['data'][idxs[:num_train]], dtype=np.float32)
    #    y_train = np.array(cov_type['target'][idxs[:num_train]], dtype=np.int32) - 1
    #    X_test = np.array(cov_type['data'][idxs[num_train:]], dtype=np.float32)
    #    y_test = np.array(cov_type['target'][idxs[num_train:]], dtype=np.int32) - 1
    #    mkdirs("data/generated/")
    #    np.save("data/generated/X_train.npy",X_train)
    #    np.save("data/generated/X_test.npy",X_test)
    #    np.save("data/generated/y_train.npy",y_train)
    #    np.save("data/generated/y_test.npy",y_test)

    # elif dataset in ('rcv1', 'SUSY', 'covtype'):
    #     X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
    #     X_train = X_train.todense()
    #     num_train = int(X_train.shape[0] * 0.75)
    #     if dataset == 'covtype':
    #         y_train = y_train-1
    #     else:
    #         y_train = (y_train+1)/2
    #     idxs = np.random.permutation(X_train.shape[0])

    #     X_test = np.array(X_train[idxs[num_train:]], dtype=np.float32)
    #     y_test = np.array(y_train[idxs[num_train:]], dtype=np.int32)
    #     X_train = np.array(X_train[idxs[:num_train]], dtype=np.float32)
    #     y_train = np.array(y_train[idxs[:num_train]], dtype=np.int32)

    #     mkdirs("data/generated/")
    #     np.save("data/generated/X_train.npy",X_train)
    #     np.save("data/generated/X_test.npy",X_test)
    #     np.save("data/generated/y_train.npy",y_train)
    #     np.save("data/generated/y_test.npy",y_test)

    # elif dataset in ('a9a'):
    #     X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
    #     X_test, y_test = load_svmlight_file("../../../data/{}.t".format(dataset))
    #     X_train = X_train.todense()
    #     X_test = X_test.todense()
    #     X_test = np.c_[X_test, np.zeros((len(y_test), X_train.shape[1] - np.size(X_test[0, :])))]

    #     X_train = np.array(X_train, dtype=np.float32)
    #     X_test = np.array(X_test, dtype=np.float32)
    #     y_train = (y_train+1)/2
    #     y_test = (y_test+1)/2
    #     y_train = np.array(y_train, dtype=np.int32)
    #     y_test = np.array(y_test, dtype=np.int32)

    #     mkdirs("data/generated/")
    #     np.save("data/generated/X_train.npy",X_train)
    #     np.save("data/generated/X_test.npy",X_test)
    #     np.save("data/generated/y_train.npy",y_train)
    #     np.save("data/generated/y_test.npy",y_test)

    # print(">> partioning data")


    n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_clients)}


    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100

        N = y_train.shape[0]
        np.random.seed(2020)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_clients))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_clients <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break


        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_clients)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_clients)
                for j in range(n_clients):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(10)]
            contain=[]
            for i in range(n_clients):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_clients)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_clients):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_clients))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_clients)}
        
    elif partition == "mixed":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        times=[1 for i in range(10)]
        contain=[]
        for i in range(n_clients):
            current=[i%K]
            j=1
            while (j<2):
                ind=random.randint(0,K-1)
                if (ind not in current and times[ind]<2):
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_clients)}
        

        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_clients))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*n_train)

        for i in range(K):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)

            proportions_k = np.random.dirichlet(np.repeat(beta, 2))
            #proportions_k = np.ndarray(0,dtype=np.float64)
            #for j in range(n_clients):
            #    if i in contain[j]:
            #        proportions_k=np.append(proportions_k ,proportions[j])

            proportions_k = (np.cumsum(proportions_k)*len(idx_k)).astype(int)[:-1]

            split = np.split(idx_k, proportions_k)
            ids=0
            for j in range(n_clients):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1

    elif partition == "real" and dataset == "femnist":
        num_user = u_train.shape[0]
        user = np.zeros(num_user+1,dtype=np.int32)
        for i in range(1,num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, n_clients)
        net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(n_clients)}
        for i in range(n_clients):
            for j in batch_idxs[i]:
                net_dataidx_map[i]=np.append(net_dataidx_map[i], np.arange(user[j], user[j+1]))

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)




class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, transform_train=None, transform_test=None, noise_level=0, net_id=None, total=0, download=False):
    
    
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10', 'svhn', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            # transform_train = transforms.Compose([
            #     transforms.ToTensor(),
            #     AddGaussianNoise(0., noise_level, net_id, total)])

            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'femnist':
            # print(">> I am over here")
            dl_obj = FEMNIST
            # transform_train = transforms.Compose([
            #     transforms.ToTensor(),
            #     AddGaussianNoise(0., noise_level, net_id, total)])
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated
            # transform_train = transforms.Compose([
            #     transforms.ToTensor(),
            #     AddGaussianNoise(0., noise_level, net_id, total)])
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     AddGaussianNoise(0., noise_level, net_id, total)])

        # elif dataset == 'svhn':
        #     dl_obj = SVHN_custom
        #     transform_train = transforms.Compose([
        #         transforms.ToTensor(),
        #         AddGaussianNoise(0., noise_level, net_id, total)])
        #     transform_test = transforms.Compose([
        #         transforms.ToTensor(),
        #         AddGaussianNoise(0., noise_level, net_id, total)])


        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            # transform_train = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Lambda(lambda x: F.pad(
            #         Variable(x.unsqueeze(0), requires_grad=False),
            #         (4, 4, 4, 4), mode='reflect').data.squeeze()),
            #     transforms.ToPILImage(),
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     AddGaussianNoise(0., noise_level, net_id, total)
            # ])
            # # data prep for test set
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     AddGaussianNoise(0., noise_level, net_id, total)])

        else:
            dl_obj = Generated
            transform_train = None
            transform_test = None


        # print(f"Transform train: {transform_train}")

        # train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=download)
        
        train_ds = dl_obj(datadir, train=True, transform=transform_train, download=download)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=download)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds





def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        c_tmp = np.array(choice)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(len(choice), size=batch_size)
            for j in range(batch_size):
                idx[i][j] = c_tmp[int(idx[i][j])]

            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx

