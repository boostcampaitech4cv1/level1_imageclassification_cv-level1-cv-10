import os
import glob
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset): # for train and validation
    def __init__(self,root,data,transform=None):
        self.root = root
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        id, gender,race,age,path = self.data[idx]
        for image_name in os.listdir(os.path.join(path)):
            
        if self.transform:
            image = self.transform(image)
        pass

class TestDataset(Dataset): # for test
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

if __name__ == "__main__":
    train_dir = '/opt/ml/input/data/train'
    val_ratio = 0.3
    seed = 42
    
    train_csv = pd.read_csv(os.path.join(train_dir, 'train.csv'))
    data = train_csv.to_numpy()

    train_data,val_data = train_test_split(data,test_size=val_ratio, shuffle=True, random_state=seed)
    train_dataset = CustomDataset(train_dir, train_data)