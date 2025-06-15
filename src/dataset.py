import torch
from torch.utils.data import Dataset
import numpy as np

class CMBDataset(Dataset):
    def __init__(self, patch_file, label_file):
        self.patches = np.load(patch_file)
        self.labels = np.load(label_file)

        self.patches = torch.tensor(self.patches).unsqueeze(1).float()
        self.labels = torch.tensor(self.labels).long()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]
