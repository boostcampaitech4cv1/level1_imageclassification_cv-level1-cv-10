import os
import glob

from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import csv_preprocess


class CustomDataset(Dataset):  # for train and validation
    def __init__(self, root, data, transform=None):
        self.root = os.path.join(root, "images")
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, gen, age, age_category, mask, label, img_path = self.data[idx]
        ### 실제 데이터 다 넘김 ###
        gen, age, age_category, mask, label = (
            int(gen),
            int(age),
            int(age_category),
            int(mask),
            int(label),
        )
        path = os.path.join(self.root, img_path)
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)


class MulitaskDataset(Dataset):  # for train and validation
    def __init__(self, root, data, transform=None):
        self.root = os.path.join(root, "images")
        self.data = data
        self.transform = transform
        self.to_tensor = lambda x: torch.tensor(int(x))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, gen, age, age_category, mask, label, img_path = self.data[idx]
        ### 실제 데이터 다 넘김 ###
        gen, age_category, mask, label = map(
            self.to_tensor, (gen, age_category, mask, label)
        )
        age = torch.tensor(float(age))
        path = os.path.join(self.root, img_path)
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, label, gen, age, age_category, mask


class TestDataset(Dataset):  # for test
    def __init__(self, root, img_paths, transform=None):
        self.root = root
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        path = os.path.join(self.root, self.img_paths[index])
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


if __name__ == "__main__":
    train_dir = "/opt/ml/input/data/train"
    batch_size = 16
    val_ratio = 0.3
    seed = 42

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=256, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=(10,10)),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_csv = pd.read_csv(os.path.join(train_dir, "train.csv"))
    data = csv_preprocess(os.path.join(train_dir, "images"), train_csv)

    train_data, val_data = train_test_split(
        data, test_size=val_ratio, shuffle=True, random_state=seed
    )
    train_dataset = MulitaskDataset(train_dir, train_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for img, label, gen, age, age_category, mask in train_loader:
        print(img.shape, label.shape, gen.shape, age.shape)
        break

    # tmp = iter(train_dataset)
    # img, label = next(tmp)
