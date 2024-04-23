from torch.utils.data import Dataset
import numpy as np
import torch


class DeepSDF_Dataset(Dataset):
    """
    This class creates the DeepSDF dataset
    """

    def __init__(self, data_file):
        """
        Load the data from a .npz file with "pos" and "neg" keys
        """
        print("Loading data from file:", data_file)
        data_npz = np.load(data_file)
        positives = data_npz["pos"]
        negatives = data_npz["neg"]
        data_np = np.concatenate([positives, negatives], 0)
        self.data = torch.from_numpy(data_np).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
