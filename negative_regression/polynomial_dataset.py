import pandas as pd
from torch.utils.data import Dataset


class PolynomialDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.df = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.df.X.values)

    def __getitem__(self, idx):
        return {"X": self.df.X.values[idx], "y": self.df.y.values[idx]}

