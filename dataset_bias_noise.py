# ---------------------------------------------------------------
# This file has been modified from following sources: 
# Source:
# 1. https://github.com/NVlabs/LSGM/blob/main/util/ema.py (NVIDIA License)
# 2. https://github.com/NVlabs/denoising-diffusion-gan/blob/main/train_ddgan.py (NVIDIA License)
# ---------------------------------------------------------------

import os
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, STL10, FashionMNIST
from datasets_prep.lmdb_datasets import LMDBDataset
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist

from PIL import Image
import os.path
from torch.utils.data import ConcatDataset, Subset
import numpy as np
import copy
import random


# Image datasets
class CelebA_HQ(data.Dataset):
    '''Note: CelebA (about 200000 images) vs CelebA-HQ (30000 images)'''
    def __init__(self, root, partition_path, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        # Split train/val/test 
        self.partition_dict = {}
        self.get_partition_label(partition_path)
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.save_img_path()
        print('[Celeba-HQ Dataset]')
        print(f'Train {len(self.train_dataset)} | Val {len(self.val_dataset)} | Test {len(self.test_dataset)}')

        if mode == 'train':
            self.dataset = self.train_dataset
        elif mode == 'val':
            self.dataset = self.val_dataset
        elif mode == 'test':
            self.dataset = self.test_dataset
        else:
            raise ValueError

    def get_partition_label(self, list_eval_partition_celeba_path):
        '''Get partition labels (Train 0, Valid 1, Test 2) from CelebA
        See "celeba/Eval/list_eval_partition.txt"
        '''
        with open(list_eval_partition_celeba_path, 'r') as f:
            for line in f.readlines():
                filenum = line.split(' ')[0].split('.')[0] # Use 6-digit 'str' instead of int type
                partition_label = int(line.split(' ')[1]) # 0 (train), 1 (val), 2 (test)
                self.partition_dict[filenum] = partition_label

    def save_img_path(self):
        for filename in os.listdir(self.root):
            assert os.path.isfile(os.path.join(self.root, filename))
            filenum = filename.split('.')[0]
            label = self.partition_dict[filenum]
            if label == 0:
                self.train_dataset.append(os.path.join(self.root, filename))
            elif label == 1:
                self.val_dataset.append(os.path.join(self.root, filename))
            elif label == 2:
                self.test_dataset.append(os.path.join(self.root, filename))
            else:
                raise ValueError

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.dataset)




class AnomalyDataset(data.Dataset):
    def __init__(self, dataset, anomaly_dataset, frac=0.01):
        '''
        dataset : target dataset (CIFAR10)
        anomaly_dataset : anomaly dataset (MNIST)
        frac : fraction of anomaly dataset (p=0.01)
        '''
        try: normal_sample, _ = dataset[0]
        except: normal_sample = dataset[0]
        c, size, _ = normal_sample.shape # [c, w, h]
        
        self.dataset = dataset
        self.anomaly_dataset = anomaly_dataset

        self.num_normal = dataset.__len__()
        self.num_anomaly = int(frac * self.num_normal)
        
        self.ANOMALIES = []
        for i in range(self.num_anomaly):
            # get samples
            x = anomaly_dataset[i]
            try: x, _ = x
            except: pass
            # check if image size is same
            if i==0: assert x.shape[1] == size
            # match the number of channels
            if x.shape[0]==1 and c==3:
                x = x.repeat(3,1,1)
            # append to self.ANOMALIES
            self.ANOMALIES.append(x)
    
    def __getitem__(self, index):
        if index < self.num_normal:
            x = self.dataset[index]
            try: x, _ = x
            except: pass
        else:
            x = self.ANOMALIES[index-self.num_normal]
        
        return x

    def __len__(self):
        return self.num_normal + self.num_anomaly


def getCleanData(dataset, image_size=32):
    if dataset == 'cifar10':
        dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(image_size),  # Resize images to your desired size
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
    
    elif dataset == 'cifar10_flip':
        dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                    transforms.Resize(image_size),  # Resize images to your desired size
                    transforms.RandomVerticalFlip(p=1.0),  # Flip all images vertically
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
        
    if dataset == 'mnist_cifar10':
        datasetcifar = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(image_size),  # Resize images to your desired size
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        datasetcifar = [(img, -1) for img, label in datasetcifar if label == 0][:500]
        
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        datasetmnist = MNIST(root='./data', train=True, transform=train_transform, download=True)

        class_1_images = [(img, label) for img, label in datasetmnist if label == 1][:5000]
        class_2_images = [(img, label) for img, label in datasetmnist if label == 2][:4000]
        class_0_images = [(img, label) for img, label in datasetmnist if label == 0][:500]

        dataset = ConcatDataset([
            datasetcifar,
            class_1_images,
            class_2_images,
            class_0_images
        ])

    if dataset == 'mnist12':
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        datasetmnist = MNIST(root='./data', train=True, transform=train_transform, download=True)
        class_1_images = [(img, label) for img, label in datasetmnist if label == 1][:5000]
        class_2_images = [(img, label) for img, label in datasetmnist if label == 2][:4500]
        class_0_images = [(img, label) for img, label in datasetmnist if label == 0][:500]

        dataset = ConcatDataset([
            class_1_images,
            class_2_images,
            class_0_images
        ])

    elif dataset == 'mnist':
        
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = MNIST(root='./data', train=True, transform=train_transform, download=True)

    elif dataset == 'fashion_mnist':
        
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = FashionMNIST(root='./data', train=True, transform=train_transform, download=True)

    elif dataset == 'mnist_1c':
        
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = MNIST(root='./data', train=True, transform=train_transform, download=True)

    elif dataset == 'fashion_mnist_1c':
        
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = FashionMNIST(root='./data/fashion_mnist', train=True, transform=train_transform, download=True)

    elif dataset == 'stl10':
        dataset = STL10('./data', split="unlabeled", transform=transforms.Compose([
                        transforms.Resize(64),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)

    elif dataset == 'clipart':
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to your desired size
            transforms.CenterCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/clipart', transform=train_transform)

    elif dataset == 'quickdraw':
        
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to your desired size
            transforms.CenterCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/quickdraw', transform=train_transform)

    elif dataset == 'sketch':
        
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize images to your desired size
            transforms.CenterCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/sketch', transform=train_transform)

    elif dataset == 'sketch_64':
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/sketch_64', transform=train_transform)

    elif dataset == 'clipart_64':
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/clipart_64', transform=train_transform)

    elif dataset == 'isolation_forest_celeba_5p_fashion':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Use torchvision's ImageFolder dataset to load your custom dataset
        dataset = torchvision.datasets.ImageFolder(root='./data/isolation_forest_celeba_5p_fashion', transform=train_transform)

    elif dataset == 'stackmnist':
        train_transform, valid_transform = _data_transforms_stacked_mnist()
        dataset = StackedMNIST(root='./data', train=True, download=False, transform=train_transform)
        
    elif dataset == 'lsun':
        
        train_transform = transforms.Compose([
                        transforms.Resize((image_size, image_size)),
                        transforms.CenterCrop((image_size, image_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ])

        train_data = LSUN(root='./data/LSUN/', classes=['church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = Subset(train_data, subset)
        

    elif dataset == 'celeba_256':
        # print(image_size)
        # print(type(image_size))
        train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        
        dataset = LMDBDataset(root='./data/celeba-lmdb/', name='celeba', train=True, transform=train_transform)

    elif dataset == 'celeba_hq':
        # print(image_size)
        # print(type(image_size))
        train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        
        dataset = LMDBDataset(root='./data/celeba-lmdb/', name='celeba', train=True, transform=train_transform)

    elif dataset == 'celeba_256_flip':
        # print(image_size)
        # print(type(image_size))
        train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.RandomVerticalFlip(p=1.0),  # Flip all images vertically,
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        
        dataset = LMDBDataset(root='./data/celeba-lmdb/', name='celeba', train=True, transform=train_transform)

    elif dataset == 'cifar10_pretrained':
        # dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
        #                 transforms.Resize(image_size),
        #                 transforms.RandomHorizontalFlip(),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        # subdatasets = list()
        # for image_type in range(10):
        #     subdatasets.append([(img, label) for img, label in dataset if label == image_type][:30])
        # dataset = ConcatDataset(subdatasets)

        num_data_per_type = 30
        if os.path.exists(f'./data/cifar10_pretrained_{num_data_per_type}.pth'):
            # Load the dataset from the file
            dataset = torch.load(f'./data/cifar10_pretrained_{num_data_per_type}.pth')
        else:
            dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
            subdatasets = list()
            for image_type in range(10):
                subdatasets.append([(img, label) for img, label in dataset if label == image_type][:num_data_per_type])
            dataset = ConcatDataset(subdatasets)
            torch.save(dataset, f'./data/cifar10_pretrained_{num_data_per_type}.pth')

    elif dataset == 'mnist_pretrained':
        num_data_per_type = 30
        if os.path.exists(f'./data/mnist_pretrained_{num_data_per_type}.pth'):
            # Load the dataset from the file
            dataset = torch.load(f'./data/mnist_pretrained_{num_data_per_type}.pth')
        else:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            dataset = MNIST(root='./data', train=True, transform=train_transform, download=True)
            subdatasets = list()
            for image_type in range(10):
                subdatasets.append([(img, label) for img, label in dataset if label == image_type][:num_data_per_type])
            dataset = ConcatDataset(subdatasets)
            torch.save(dataset, f'./data/mnist_pretrained_{num_data_per_type}.pth')
        
    return dataset

def getMixedData(source_dataset, perturb_dataset, percentage = 0, image_size=32, random_seed = 19, shuffle = False):
    random.seed(random_seed)
    name_source = source_dataset
    source_dataset = getCleanData(source_dataset, image_size=image_size)
    perturb_dataset = getCleanData(perturb_dataset, image_size=image_size)
    if name_source in ['sketch', 'sketch_64']:
        num_samples = 30000
    else:
        num_samples = len(source_dataset)
    
    print(f'number of samples: {num_samples}')
    num_perturbed_samples = int(int(num_samples) * percentage / 100)
    
    num_source_samples = num_samples - num_perturbed_samples

    source_indices = random.sample(range(len(source_dataset)), num_source_samples)  # Randomly select indices of source data
    perturbed_indices = random.sample(range(len(perturb_dataset)), num_perturbed_samples)  # Randomly select indices of perturbed data

    print(f'source has {num_source_samples} data')
    print(f'perturb has {num_perturbed_samples} data')

    dataset = ConcatDataset([
        Subset(source_dataset, source_indices),
        Subset(perturb_dataset, perturbed_indices)
    ])

    if shuffle:
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        # Create a new dataset with shuffled indices
        dataset = Subset(dataset, indices)

    return dataset

def getNewMixedData(source_dataset, perturb_dataset, cpu_mean_image, percentage = 0, image_size=32, random_seed = 19, shuffle = False):
    random.seed(random_seed)
    name_source = source_dataset
    source_dataset = getCleanData(source_dataset, image_size=image_size)
    perturb_dataset = getCleanData(perturb_dataset, image_size=image_size)

    batch_mean_images = cpu_mean_image.unsqueeze(0).repeat(len(source_dataset), 1, 1, 1).permute(0, 2, 3, 1).numpy()
    source_dataset.data = source_dataset.data - batch_mean_images
    print('finish source dataset')
    import ipdb; ipdb.set_trace()
    batch_mean_images = cpu_mean_image.unsqueeze(0).repeat(len(perturb_dataset), 1, 1, 1).permute(0, 2, 3, 1).numpy()
    perturb_dataset.data = perturb_dataset.data - batch_mean_images
    print('finish perturb dataset')

    if name_source in ['sketch', 'sketch_64']:
        num_samples = 30000
    else:
        num_samples = len(source_dataset)
    
    print(f'number of samples: {num_samples}')
    num_perturbed_samples = int(int(num_samples) * percentage / 100)
    
    num_source_samples = num_samples - num_perturbed_samples

    source_indices = random.sample(range(len(source_dataset)), num_source_samples)  # Randomly select indices of source data
    perturbed_indices = random.sample(range(len(perturb_dataset)), num_perturbed_samples)  # Randomly select indices of perturbed data

    print(f'source has {num_source_samples} data')
    print(f'perturb has {num_perturbed_samples} data')

    dataset = ConcatDataset([
        Subset(source_dataset, source_indices),
        Subset(perturb_dataset, perturbed_indices)
    ])

    if shuffle:
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        # Create a new dataset with shuffled indices
        dataset = Subset(dataset, indices)

    return dataset


# ------------------------
# For Toy
# ------------------------
# datasets
class ToydatasetGaussian(data.Dataset):
    def __init__(self, cfg):
        self.dataset = torch.randn(cfg.num_data, cfg.data_dim) + torch.tensor([0,10])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class Toydatasetp(data.Dataset):
    def __init__(self, cfg):
        std = 0.5
        self.dataset = torch.cat([std*torch.randn(cfg.num_data//2, cfg.data_dim)+1, 
                                  std*torch.randn(cfg.num_data-cfg.num_data//2, cfg.data_dim)-1])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class Toydatasetq(data.Dataset):
    def __init__(self, cfg):
        std = 0.5
        self.dataset = torch.cat([std*torch.randn(2*cfg.num_data//3, cfg.data_dim)+2, 
                                  std*torch.randn(cfg.num_data-2*cfg.num_data//3, cfg.data_dim)-1])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class ToydatasetOutlier(data.Dataset):
    def __init__(self, cfg):
        M = int(cfg.num_data*cfg.p)
        self.dataset = torch.cat([0.1*torch.randn(cfg.num_data-M, cfg.data_dim) + 1, 0.1*torch.randn(M, cfg.data_dim) - 1])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class ToydatasetNoise(data.Dataset):
    def __init__(self, cfg):
        self.N = cfg.num_data
        self.dim = cfg.data_dim
    
    def __len__(self):
        return int(self.N)
        
    
    def __getitem__(self, idx):
        return torch.randn((1, self.dim))


def get_datasets(cfg):
    src_name, tar_name = cfg.source_name, cfg.target_name
    datasets = []

    for name in [src_name, tar_name]:
        if name == 'gaussian':
            dataset = ToydatasetGaussian(cfg)
        elif name == 'p':
            dataset = Toydatasetp(cfg)
        elif name == 'q':
            dataset = Toydatasetq(cfg)
        elif name == 'outlier':
            dataset = ToydatasetOutlier(cfg)
        elif name == 'noise':
            dataset = ToydatasetNoise(cfg)
        else:
            raise NotImplementedError
        
        datasets.append(dataset)
    
    return datasets
