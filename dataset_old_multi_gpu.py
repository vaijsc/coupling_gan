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
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, STL10, FashionMNIST
from datasets_prep.lmdb_datasets import LMDBDataset
from PIL import Image
import os.path
from torch.utils.data import ConcatDataset, Subset
import numpy as np
import copy


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


# get dataloader
def get_dataset(args):
    if args.dataset == 'mnist':
        dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)
        # dataset = Subset(dataset, list(range(64)))

    elif args.dataset == 'stl10':
        dataset = STL10('./data', split="unlabeled", transform=transforms.Compose([
                        transforms.Resize(64),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
    
    elif args.dataset == 'cifar10':
        dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        # subset_indices = list(range(64))
        # dataset = Subset(dataset, subset_indices) 
        # dataset = Subset(dataset, list(range(64)))

    elif args.dataset == 'cifar10_pretrained':
        # dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
        #                 transforms.Resize(args.image_size),
        #                 transforms.RandomHorizontalFlip(),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        # subdatasets = list()
        # for image_type in range(10):
        #     subdatasets.append([(img, label) for img, label in dataset if label == image_type][:30])
        # dataset = ConcatDataset(subdatasets)

        num_data_per_type = 20
        if os.path.exists(f'./data/cifar10_pretrained_{num_data_per_type}.pth'):
            # Load the dataset from the file
            dataset = torch.load(f'./data/cifar10_pretrained_{num_data_per_type}.pth')
        else:
            dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                            transforms.Resize(args.image_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
            subdatasets = list()
            for image_type in range(10):
                subdatasets.append([(img, label) for img, label in dataset if label == image_type][:num_data_per_type])
            dataset = ConcatDataset(subdatasets)
            torch.save(dataset, f'./data/cifar10_pretrained_{num_data_per_type}.pth')

    elif args.dataset == 'mnist_pretrained':
        
        # train_transform = transforms.Compose([
        #     transforms.Resize(args.image_size),
        #     transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        # dataset = MNIST(root='./data', train=True, transform=train_transform, download=True)
        # subdatasets = list()
        # for image_type in range(10):
        #     subdatasets.append([(img, label) for img, label in dataset if label == image_type][:30])
        # dataset = ConcatDataset(subdatasets)

        num_data_per_type = 20
        if os.path.exists(f'./data/mnist_pretrained_{num_data_per_type}.pth'):
            # Load the dataset from the file
            dataset = torch.load(f'./data/mnist_pretrained_{num_data_per_type}.pth')
        else:
            train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
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
    
    elif args.dataset == 'cifar10+mnist':
        normal_dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        
        anomaly_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset)

    elif args.dataset == 'cifar10+3mnist':
        normal_dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        
        anomaly_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.03)

    elif args.dataset == 'cifar10+5mnist':
        normal_dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        
        anomaly_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.05)

    elif args.dataset == 'mnist+cifar10':
        normal_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)

        anomaly_dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)                
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.01)

    elif args.dataset == 'mnist+3cifar10':
        normal_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)

        anomaly_dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)                
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.03)

    elif args.dataset == 'mnist+5cifar10':
        normal_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)

        anomaly_dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)                
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.05)

    elif args.dataset == 'fashion_mnist_1p_mnist':
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        normal_dataset = FashionMNIST(root='./data/fashion_mnist', train=True, transform=train_transform, download=True)
        
        anomaly_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.01)

    elif args.dataset == 'fashion_mnist_3p_mnist':
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        normal_dataset = FashionMNIST(root='./data/fashion_mnist', train=True, transform=train_transform, download=True)
        
        anomaly_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.03)

    elif args.dataset == 'fashion_mnist_5p_mnist':
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        normal_dataset = FashionMNIST(root='./data/fashion_mnist', train=True, transform=train_transform, download=True)
        
        anomaly_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.05)

    elif args.dataset == 'mnist_1p_fashion_mnist':
        normal_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)

        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        anomaly_dataset = FashionMNIST(root='./data/fashion_mnist', train=True, transform=train_transform, download=True)
        
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.01)

    elif args.dataset == 'mnist_3p_fashion_mnist':
        normal_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)

        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        anomaly_dataset = FashionMNIST(root='./data/fashion_mnist', train=True, transform=train_transform, download=True)
        
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.03)

    elif args.dataset == 'mnist_5p_fashion_mnist':
        normal_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)

        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        anomaly_dataset = FashionMNIST(root='./data/fashion_mnist', train=True, transform=train_transform, download=True)
        
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.05)

    elif args.dataset == 'celeba_64_5p_fashion':
        # print(image_size)
        # print(type(image_size))
        train_transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.CenterCrop((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        
        normal_dataset = LMDBDataset(root='./data/celeba-lmdb/', name='celeba', train=True, transform=train_transform)

        train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.Grayscale(num_output_channels=3),  # Convert to "RGB" format (with duplicated channels)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        anomaly_dataset = FashionMNIST(root='./data/fashion_mnist', train=True, transform=train_transform, download=True)

        dataset = AnomalyDataset(normal_dataset, anomaly_dataset, frac = 0.05)

    
    elif args.dataset == 'celeba_256':
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = CelebA_HQ(
            root='data/celeba-hq/celeba-256',
            partition_path='data/celeba-hq/list_eval_partition_celeba.txt',
            mode='train', # 'train', 'val', 'test'
            transform=train_transform,
        )
    
    else: NotImplementedError
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
