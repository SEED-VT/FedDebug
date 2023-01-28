import torchvision
import torch
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, random_split
# from torch.utils.data import Dataset
import random
import numpy as np
# from .niid_datasets import load_femnist_data
from .niid_utils import partition_data, get_dataloader


class PrepareDatasetFromXY(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        x, y = self.X[index], self.y[index]
        return x, y

    def __len__(self):
        return len(self.y)


def initializeTrainAndValidationDataset(dataset_name, data_dir):

    resize_trasnform = torchvision.transforms.Resize((32, 32))

    if dataset_name == "cifar10":
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transforms)

        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transforms)

        return train_dataset, val_dataset, 10

    elif dataset_name == "femnist":

        transform = torchvision.transforms.Compose([resize_trasnform,
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                    ])

        # try:
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset_name, data_dir, data_dir, "homo", 10, beta=0.1, download=True, transform=None)
        num_classes = len(np.unique(y_train))
        # except:
        #     X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        #         dataset_name, data_dir, data_dir, "homo", 10, beta=0.1, download=False, transform=None)
        #     num_classes = len(np.unique(y_train))
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        dataidxs = net_dataidx_map[0]
        _, _, train_ds, test_ds = get_dataloader(
            dataset_name, data_dir, 256, 256, dataidxs, transform_train=transform, transform_test=transform)

        train_ds = SubsetToDataset(train_ds)
        valid_ds = SubsetToDataset(test_ds)

        return train_ds, valid_ds, num_classes

    else:
        raise ValueError("Dataset name not recognized")


# def getLabel2Data(data_set,  num_classes=10):
#     labels2indexes = {i: [] for i in range(num_classes)}
#     _ = [labels2indexes[data_set[i][1]].append(
#         i) for i in range(len(data_set))]
#     label2data = {l: torch.utils.data.Subset(
#         data_set, labels2indexes[l]) for l in range(num_classes)}
#     return label2data


# def splitDataSetIntoNClientsIID(train, clients, num_classes=10):
#     def split(dataset):
#         parts = [len(dataset)//clients for _ in range(clients)]
#         parts[0] += len(dataset) % clients
#         subsets = torch.utils.data.random_split(dataset, parts)
#         return [SubsetToDataset(subset) for subset in subsets]

#     label2data = getLabel2Data(train, num_classes=num_classes)
#     labe2clientsdata = {}
#     for i in range(num_classes):
#         labe2clientsdata[i] = split(label2data[i])
    
#         # print(f"Spliting Datasets {len(dataset)} into parts:{parts}")


#     return [torch.utils.data.ConcatDataset([labe2clientsdata[lid][pid] for lid in range(10)]) for pid in range(clients)]


# # use https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
#     # def generateRandomWeights(n):
#     # to randomly assing uneven data to each client

# def splitDataSetIntoNClientsNonIID(train, clients, num_classes=10 ,beta=3):
#     def split(dataset):
#         assert beta > 0 and beta < 4, "beta must be in (0,4)"
#         # n = len(clients)
#         min_data = (len(dataset)//clients)//beta
#         max_data = len(dataset)//clients
#         parts = [random.randint(min_data, max_data) for _ in range(clients)]
#         remaining_data = len(dataset) - sum(parts)
#         i = 0
#         while remaining_data > 0:
#             parts[i] += 1
#             remaining_data -= 1
#             i += 1
#             if i == len(parts):
#                 i = 0

#         assert sum(parts) == len(
#             dataset), f"Sum of parts is not equal to dataset size: {sum(parts)} != {len(dataset)}"

#         # print(f"Spliting Datasets {len(dataset)} into parts:{parts}")
#         subsets = torch.utils.data.random_split(dataset, parts)
#         clients_datasets = [SubsetToDataset(subset) for subset in subsets]
#         assert sum(parts) == sum([len(d) for d in clients_datasets]
#                                  ), f"Sum of parts is not equal to dataset size: {sum(parts)} != {sum([len(dataset) for dataset in clients_datasets])}"
#         return clients_datasets
    
#     label2data = getLabel2Data(train, num_classes=num_classes)
#     labe2clientsdata = {}
#     for i in range(num_classes):
#         labe2clientsdata[i] = split(label2data[i])

#     return [torch.utils.data.ConcatDataset([labe2clientsdata[lid][pid] for lid in range(10)]) for pid in range(clients)]


def splitDataSetIntoNClientsIID(dataset, clients):

    parts = [len(dataset)//clients for _ in range(clients)]

    parts[0] += len(dataset) % clients

    print(f"Spliting Datasets {len(dataset)} into parts:{parts}")


    subsets =  torch.utils.data.random_split(dataset, parts)

    return [SubsetToDataset(subset) for subset in subsets]


# use https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
    # def generateRandomWeights(n):
    # to randomly assing uneven data to each client

def splitDataSetIntoNClientsNonIID(dataset, clients, beta=3):
    
    assert beta > 0 and beta < 4, "beta must be in (0,4)"

    # n = len(clients)
    min_data =  (len(dataset)//clients)//beta
    max_data =  len(dataset)//clients

    parts = [random.randint(min_data, max_data) for _ in range(clients)]

    # print(f">> Before Distributing the mode amount: Spliting Datasets {len(dataset)} into parts:{parts}")

    # for r in range(len(parts)%sum(parts)):
    remaining_data = len(dataset) - sum(parts)
    i = 0
    while remaining_data > 0:
        parts[i] += 1
        remaining_data -= 1
        i += 1
        if i == len(parts):
            i = 0
    
    assert sum(parts) == len(dataset) , f"Sum of parts is not equal to dataset size: {sum(parts)} != {len(dataset)}"

    # parts2 = [np ]

    # parts = [len(dataset)//n for _ in range(n)]
    # parts[0] += len(dataset) % n
    # print(f">> After Distributing the mode amount: Spliting Datasets {len(dataset)} into parts:{parts}")
    print(f"Spliting Datasets {len(dataset)} into parts:{parts}")
    subsets =  torch.utils.data.random_split(dataset, parts)
    clients_datasets = [SubsetToDataset(subset) for subset in subsets]

    assert sum(parts) == sum([len(d) for d in clients_datasets]), f"Sum of parts is not equal to dataset size: {sum(parts)} != {sum([len(dataset) for dataset in clients_datasets])}"

    return clients_datasets



class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes, noise_rate):
        assert noise_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], f"Noise rate must be in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] but got {noise_rate}"
        self.dataset = dataset
        self.num_classes = num_classes
        self.class_ids = random.sample(range(num_classes), int(noise_rate*num_classes))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if y in self.class_ids:
            y_hat = random.randint(0, self.num_classes-1)
            if y_hat != y:
                y = y_hat
            else:
                y = (y+1)%self.num_classes
        return x, y


class AttackBackdoor(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes, class_ids_to_poison, attack_pattern, backdoor_target_class_id):
        self.dataset = dataset
        self.class_ids = class_ids_to_poison
        self.num_classes = num_classes
        self.attack_pattern = attack_pattern
        self.target_class_id = backdoor_target_class_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if y in self.class_ids and idx % 2 == 0:
            y = self.target_class_id
            x += self.attack_pattern
        return x, y


class SubsetToDataset(torch.utils.data.Dataset):
    def __init__(self, subset, greyscale=False):
        self.subset = subset
        # self.X, self.Y = self.subset, self.subset.target
        self.greyscale = greyscale

    def __getitem__(self, index):
        x, y = self.subset[index]
        # print(">> XShape", x.shape, y)
        # trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        # x = trans(x)
        # import torchvision.transforms as T
        # x =  T.Grayscale(num_output_channels=3)(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class FedDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, num_workers=4) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.drop_last = False
        if len(self.train_dataset) % self.batch_size == 1:
            self.drop_last = True
            print(
                f"Dropping last batch because of uneven data size: {len(self.train_dataset)} % {self.batch_size} == 1")

        # print(
        #     f"Train mod batch = {len(train_dataset) % batch_size}, and drop_last = {self.drop_last}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
