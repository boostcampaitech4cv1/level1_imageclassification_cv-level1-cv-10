import os
import pandas as pd
from loss import MultitaskLoss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm
from sklearn.model_selection import train_test_split

from dataset import MulitaskDataset
from utils import csv_preprocess, build_transform, make_class


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


class MultitaskHead(nn.Module):
    def __init__(self, in_features):
        super(MultitaskHead, self).__init__()
        self.gen_head = nn.Linear(in_features=in_features, out_features=2)
        self.age_head = nn.Linear(in_features=in_features, out_features=3)
        self.mask_head = nn.Linear(in_features=in_features, out_features=3)

    def forward(self, x):
        gen_pred = self.gen_head(x)
        age_pred = self.age_head(x)
        mask_pred = self.mask_head(x)
        return gen_pred, age_pred, mask_pred


class MultitaskModel(nn.Module):
    def __init__(self, args=None):
        super(MultitaskModel, self).__init__()
        if args.backbone_name == "resnet50":
            self.backbone = timm.create_model(
                "resnet50", pretrained=True, num_classes=0
            )
            self.head = MultitaskHead(in_features=2048)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    from collections import namedtuple

    args_template = namedtuple("args_template", ["backbone_name"])
    args = args_template("resnet50")

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
    train_dataset = MulitaskDataset(train_dir, train_data, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = MultitaskLoss(args).cuda()
    model = MultitaskModel(args).cuda()

    for img, label, gen, age, age_category, mask in train_loader:
        img, label, gen, age, age_category, mask = (
            img.cuda(),
            label.cuda(),
            gen.cuda(),
            age.cuda(),
            age_category.cuda(),
            mask.cuda(),
        )
        gen_pred, age_pred, mask_pred = model(img)
        gen_loss, age_loss, mask_loss = loss_fn(
            gen_pred, age_pred, mask_pred, gen, age_category, mask
        )
        print(make_class(gen_pred, age_pred, mask_pred))
        break
