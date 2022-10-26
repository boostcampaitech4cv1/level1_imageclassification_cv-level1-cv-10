import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm
from sklearn.model_selection import train_test_split

from dataset import CustomDataset
from utils import csv_preprocess, build_transform


class CustomModel(nn.Module):
    def __init__(self, args=None):
        super(CustomModel, self).__init__()
        if args.backbone_name == "resnet50":
            self.backbone = timm.create_model(
                "resnet50", pretrained=True, num_classes=18
            )

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == "__main__":
    train_dir = "/opt/ml/input/data/train"
    batch_size = 16
    val_ratio = 0.3
    seed = 42

    train_csv = pd.read_csv(os.path.join(train_dir, "train.csv"))
    data = csv_preprocess(os.path.join(train_dir, "images"), train_csv)

    train_transform, _ = build_transform(None, "train")
    train_data, val_data = train_test_split(
        data, test_size=val_ratio, shuffle=True, random_state=seed
    )
    train_dataset = CustomDataset(train_dir, train_data, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    model = CustomModel().cuda()

    for img, label in train_loader:
        img, label = img.cuda(), label.cuda()
        pred = model(img)
        loss = loss_fn(pred, label)
        break
