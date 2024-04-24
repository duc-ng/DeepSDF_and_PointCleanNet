from torch.utils.data import Dataset
import numpy as np
import torch
from glob import glob


class DeepSDF_Dataset(Dataset):
    """
    This class creates the DeepSDF dataset
    """

    def __init__(self):
        """
        Load the data from a .npz file with "pos" and "neg" keys
        """
        self.input_dir = "out/1_preprocessed"
        input_files = glob(self.input_dir + "/*.npz")
        # init torch tensor with size of the number of input files
        self.data = []
        print("Loading data from directory:", self.input_dir)
        for input_file in input_files:
            data_npz = np.load(input_file)
            positives = data_npz["pos"]
            negatives = data_npz["neg"]
            data_np = np.concatenate([positives, negatives], 0)
            data_torch = torch.from_numpy(data_np).float()
            self.data.append(data_torch)
        self.data = torch.stack(self.data).permute(1, 0, 2) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], index
