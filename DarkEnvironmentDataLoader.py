import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random

class DarkEnvironmentDataLoader(data.Dataset):
    def __init__(self, dark_environment_images_path):
        self.train_list = self.populate_train_list(dark_environment_images_path)
        self.size = 256
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def populate_train_list(self, dark_environment_images_path):
        image_list = glob.glob(dark_environment_images_path + "*.jpg")
        train_list = image_list
        random.shuffle(train_list)
        return train_list

    def __getitem__(self, index):
        data_path = self.data_list[index]
        data = Image.open(data_path)
        data = data.resize((self.size, self.size), Image.ANTIALIAS)
        data = (np.asarray(data) / 255.0)
        data = torch.from_numpy(data).float()
        return data.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)
