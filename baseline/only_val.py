import os
import argparse
import random
from datetime import datetime
from datetime import datetime
from pytz import timezone

import wandb
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from utils import *
from metric import accuracy, macro_f1
from dataset import MulitaskDataset
from model import CustomModel, MultitaskModel
from process import train, validation

from loss import MultitaskLoss, create_criterion

"""
TODO
hyperparameter tunning tool (ray tune or optuna or )
TTA
data preprocess (ex. background subtraction)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.3)  # train-val slit ratio
    parser.add_argument("--stratify", type=bool, default=True)
    parser.add_argument("--age_pred", type=str, default="ordinary")

    # parser.add_argument("--in_size", type=int, default=224) # input size image
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_workers", type=int, default=4)

    # parser.add_argument("--print_iter", type=int, default=10)
    # parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--train_dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--save_dir", type=str, default="/opt/ml/experiment/")
    parser.add_argument("--backbone_name", type=str, default="resnet50")
    parser.add_argument("--project_name", type=str, default="multitask")
    parser.add_argument("--experiment_name", type=str, default="age regression")
    args = parser.parse_args()

    # wandb.init(project=args.project_name, name=args.experiment_name, entity="cv-10")
    # wandb.config.update(args)

    set_seed(args.seed)

    # save_path = os.path.join(args.save_dir, args.project_name, args.experiment_name)+datetime.now(timezone('Asia/Seoul')).strftime("(%m.%d %H:%M)")
    # os.makedirs(save_path, exist_ok=False)

    train_csv = pd.read_csv(os.path.join(args.train_dir, "train.csv"))
    data = csv_preprocess(os.path.join(args.train_dir, "images"), train_csv)
    if args.stratify:
        train_data, val_data = train_test_split(
            data,
            test_size=args.val_ratio,
            shuffle=True,
            random_state=args.seed,
            stratify=data[:, 3],  ## age_class stratify
        )
    else:
        train_data, val_data = train_test_split(
            data,
            test_size=args.val_ratio,
            shuffle=True,
            random_state=args.seed,  ## no stratify
        )
    train_transform, val_transform = build_transform(
        args=args, phase="train"
    )  # data augmentation
    train_dataset = MulitaskDataset(
        args.train_dir, train_data, transform=train_transform
    )
    val_dataset = MulitaskDataset(args.train_dir, val_data, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    loss_fn = MultitaskLoss(args).cuda()
    model = MultitaskModel(args).cuda()

    ### load model ###
    ckpt = torch.load(
        "/opt/ml/experiment/multitask/age classification - stratify(10.30 12:52)/model_28.pth"
    )
    model.load_state_dict(ckpt["model"])
    print("LOADED model")

    ### validation ###
    validation(
        args=args,
        epoch=-1,
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
    )
