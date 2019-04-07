from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

# class ChairDataset(Dataset):
#     def __init__(self, csv_file, root_dir):
#         self.frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#
#     def __len__(self):
#         return len(self.frame)
#
#     def __getitem__(self, idx):
#         file_name = self.root_dir + self.frame.iloc[idx, 1]
#
#         image3D = np.load(file_name)
#
#         # todo: normalize dataset according to dataset structure
#         image3D /= 255
#         image3D = torch.from_numpy(image3D)
#         return image3D

class ChairDataset(Dataset):
    def __init__(self, root_dir, len):
        self.len = len
        self.root_dir = root_dir

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_name = self.root_dir + str(idx+1) + ".npy"

        image3D = np.load(file_name)
        image3D = image3D.astype(float)
        # todo: normalize dataset according to dataset structure
        image3D = torch.from_numpy(image3D)
        # print(image3D)
        # print(image3D.unique())
        return image3D