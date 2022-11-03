import os
import pandas as pd
from loss import MultitaskLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, args, in_features, embed_dim):
        super(MultitaskHead, self).__init__()

        # self.gen_head = nn.Sequential(
        #     nn.Linear(in_features=in_features, out_features=embed_dim),
        #     nn.GELU(),
        #     nn.Dropout(args.dropout),
        #     nn.Linear(in_features=embed_dim, out_features=2),
        #     )
        
        # self.mask_head = nn.Sequential(
        #     nn.Linear(in_features=in_features, out_features=embed_dim),
        #     nn.GELU(),
        #     nn.Dropout(args.dropout),
        #     nn.Linear(in_features=embed_dim, out_features=3),
        #     )

        self.gen_head = nn.Linear(in_features=in_features, out_features=2)
        self.mask_head = nn.Linear(in_features=in_features, out_features=3)

        if args.age_pred == 'classification':
            if args.activation_head:
                print('activation_head')      
                self.age_head = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=embed_dim),
                    nn.GELU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(in_features=embed_dim, out_features=3),
                )
            else:
                self.age_head = nn.Linear(in_features=in_features, out_features=3)
        elif args.age_pred == 'regression' or args.age_pred == 'cls_regression':
            self.age_head = nn.Linear(in_features=in_features, out_features=1)
        elif args.age_pred == 'ordinary':
            self.age_head = nn.Linear(in_features=in_features, out_features=1)


    def forward(self, x):
        gen_pred = self.gen_head(x)
        age_pred = self.age_head(x)
        # age_pred = 2*F.sigmoid(self.age_head(x)) # for 0~2, logistic regression => binary classification에 가까워질 수 도?
        mask_pred = self.mask_head(x)
        return gen_pred, age_pred, mask_pred


class MultitaskModel(nn.Module):
    def __init__(self, args=None):
        super(MultitaskModel, self).__init__()
        if args.backbone_name == "resnet50":
            self.backbone = timm.create_model(
                "resnet50", pretrained=True, num_classes=0
            )
            self.head = MultitaskHead(args, in_features=2048, embed_dim=128)

        elif args.backbone_name == 'convnext_small':
            self.backbone = timm.create_model('convnext_small', pretrained=True,num_classes=0)
            self.head = MultitaskHead(args, in_features=768, embed_dim=128)

        # elif self.args.backbone_name == 'efficientnetv2':
        #     self.backbone = timm.create_model('efficientnetv2_rw_m', pretrained=True,num_classes=1)
        # elif self.args.backbone_name == 'efficientnet':
        #     self.backbone = timm.create_model('tf_efficientnet_b7', pretrained=True,num_classes=1)
        # elif self.args.backbone_name == 'convnext':
        #     self.backbone = timm.create_model('convnext_small', pretrained=True,num_classes=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    from collections import namedtuple

    args_template = namedtuple("args_template", ["backbone_name"])
    args = args_template("convnext_small")

    # print(set([name.split("_")[0] for name in timm.list_models(pretrained=True)]))
    # print()

    # target_model_list = ['vgg','inception','efficientnet','convnext','vit','swin','coat']

    # name = "resnet50"
    # tmp = timm.create_model(
    #             name, pretrained=False, num_classes=0
    #         )
    # print(name,":",sum(p.numel() for p in tmp.parameters()))

    # for name in timm.list_models(pretrained=True):
    #     if 'convnext' in name:
    #         tmp = timm.create_model(
    #             name, pretrained=False, num_classes=0
    #         )
    #         print(name,"",sum(p.numel() for p in tmp.parameters()))

    # convnext_base  87566464
    # convnext_small  49454688
    # convnext_tiny  27820128

    # swin_base_patch4_window7_224  86743224
    # swin_small_patch4_window7_224  48837258
    # swin_tiny_patch4_window7_224  27519354

    # swinv2_small_window8_256  48959418
    # swinv2_base_window8_256  86893816
    # swinv2_tiny_window8_256  27578154



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
