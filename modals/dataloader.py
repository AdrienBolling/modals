"""
This file implements a base loader interface as well as a loader specific to data from the UCR Archive.
"""
import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch

class Loader:

    def read_data(self, *args, **kwargs):
        """
        Read data from a source.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError

    def get_dataset(self, *args, **kwargs):
        """
        Get a torch dataset from the data.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError

    def get_dataloader(self, *args, **kwargs):
        """
        Get a torch dataloader from the data.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError

class UCRDataset(Dataset):

        def __init__(self, data, num_classes, transform=None):
            """
            Initialize a UCRDataset.
            Args:
                data: The data to be used in the dataset
                num_classes: The number of classes in the dataset
                transform: A transform to be applied to the data

            """
            self.data = data[:, 1:]
            raw_labels = data[:, 0].astype(int)
            # Convert the labels to a one-hot encoding
            self.labels = F.one_hot(torch.tensor(raw_labels), num_classes).numpy()

            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]

            if self.transform:
                sample = self.transform(sample)

            return sample, self.labels[idx]


class UCRLoader(Loader):

    def read_data(self, ucr_root, dataset_name):
        """
        Read data from the UCR Archive.
        Args:
            ucr_root:
            dataset_name:

        Returns:
            a tuple containing the training data, the testing data, the number of classes, and the number of features
        """
        data_path = os.path.join(ucr_root, dataset_name, dataset_name)

        train = np.loadtxt(data_path + '_TRAIN.tsv', delimiter='\t')
        test = np.loadtxt(data_path + '_TEST.tsv', delimiter='\t')

        # Get the number of classes
        classes = np.unique(np.concatenate((train[:, 0], test[:, 0]))).astype(int)
        num_classes = len(classes)

        # Get the number of features
        num_features = train.shape[1] - 1

        return train, test, num_classes, num_features

    def get_dataset(self, ucr_root, dataset_name, train_ratio=0.8):
        """
        Get a torch dataset from the UCR Archive.
        Args:
            ucr_root:
            dataset_name:
            train_ratio:

        Returns:

        """
        train, test, num_classes, num_features = self.read_data(ucr_root, dataset_name)

        # Split the training data into training and validation with train_test_split
        np.random.shuffle(train)
        split_idx = int(train_ratio * len(train))
        train_data = train[:split_idx]
        val_data = train[split_idx:]

        train_dataset = UCRDataset(train_data, num_classes)
        val_dataset = UCRDataset(val_data, num_classes)
        test_dataset = UCRDataset(test, num_classes)

        return train_dataset, val_dataset, test_dataset, num_classes, num_features

    def get_dataloader(self, ucr_root, dataset_name, train_ratio=0.8, batch_size=32, num_workers=0):

        train_dataset, val_dataset, test_dataset, num_classes, num_features = self.get_dataset(ucr_root, dataset_name, train_ratio)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader, num_classes, num_features
