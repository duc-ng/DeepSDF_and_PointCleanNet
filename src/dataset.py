from torch.utils.data import Dataset
import numpy as np
import torch
from glob import glob
import os


class DeepSDF_Dataset(Dataset):
    """
    This class creates the DeepSDF dataset
    """

    def __init__(self, nr_rand_samples):
        """
        Load the data from a .npz file 
        """
        self.input_dir = "out/1_preprocessed"
        self.nr_rand_samples = nr_rand_samples
        input_files = sorted(glob(os.path.join(self.input_dir, "*.npz")))
        self.data = []
        self.sdfs = []
        print("Loading data from directory:", self.input_dir)
        for input_file in input_files:
            data_npz = np.load(input_file)
            samples = data_npz["samples"]
            sdf = data_npz["sdf"]
            self.data.append(torch.from_numpy(samples).float())
            self.sdfs.append(torch.from_numpy(sdf).float())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx = torch.randperm(self.data[index].shape[0])[:self.nr_rand_samples]
        return self.data[index][idx], self.sdfs[index][idx], index


class SingleShape_Dataset(Dataset):
    """
    This class creates a dataset with only one shape.
    Returns the vertices and sdf value 
    """
    def __init__(self, vertices, sdfs):
        self.data = vertices
        self.sdfs = sdfs
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.sdfs[index]