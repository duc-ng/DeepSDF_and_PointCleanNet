from torch.nn.utils.parametrizations import weight_norm
import torch.nn as nn
import torch


class AutoDecoder(nn.Module):
    """
    AutoDecoder network for DeepSDF.
    """

    def __init__(self, latent_size, p=0.1):
        super(AutoDecoder, self).__init__()
        self.size = 3 + latent_size
        self.fc1 = weight_norm(nn.Linear(self.size, 512))
        self.fc2 = weight_norm(nn.Linear(512, 512))
        self.fc3 = weight_norm(nn.Linear(512, 512))
        self.fc4 = weight_norm(nn.Linear(512, 512 - self.size))
        self.fc5 = weight_norm(nn.Linear(512, 512))
        self.fc6 = weight_norm(nn.Linear(512, 512))
        self.fc7 = weight_norm(nn.Linear(512, 512))
        self.fc8 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.tanh = nn.Tanh()
        print("AutoDecoder initialized")

    def forward(self, xyz_latent):
        x = self.dropout(self.relu(self.fc1(xyz_latent)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = torch.hstack((x, xyz_latent))
        x = self.dropout(self.relu(self.fc5(x)))
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        x = self.fc8(x)
        x = self.tanh(x)
        return x
