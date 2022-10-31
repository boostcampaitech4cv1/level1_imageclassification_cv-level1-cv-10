from collections import namedtuple
import os
import argparse
import random
from datetime import datetime
from pytz import timezone

import wandb
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
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
data preprocess or augmentation (ex. background subtraction)
backbone test
psuedo label
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.3)  # train-val slit ratio
    parser.add_argument("--split_option", type=str, default="none")  # different or none
    parser.add_argument("--stratify", type=bool, default=False)
    parser.add_argument("--wrs", type=bool, default=True)

    parser.add_argument("--age_pred", type=str, default="classification") # classification or regression or ordinary
    parser.add_argument("--age_normalized", type=str, default="normal") # normal or minmax

    parser.add_argument("--in_size", type=int, default=224) # input size image
    parser.add_argument("--crop_type", type=str, default="center") # crop type : center or random or random_resized
    parser.add_argument("--degrees", type=int, default=10) # rotation degree
    parser.add_argument("--translate", type=float, default=0.1) # translate ratio

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_workers", type=int, default=4)

    parser.add_argument("--train_dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--save_dir", type=str, default="/opt/ml/experiment/")
    parser.add_argument("--backbone_name", type=str, default="resnet50")
    parser.add_argument("--project_name", type=str, default="augmentation_test")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="classification - wrs&original dataset",
    )
    args = parser.parse_args()
    age_stat = None # for regression only

    set_seed(args.seed)

    save_path = os.path.join(
        args.save_dir, args.project_name, args.experiment_name
    ) + datetime.now(timezone("Asia/Seoul")).strftime("(%m.%d %H:%M)")
    wandb.init(project=args.project_name, name=args.experiment_name, entity="cv-10")
    wandb.config.update(args)
    os.makedirs(save_path, exist_ok=False)

    all_csv = pd.read_csv(os.path.join(args.train_dir, "train.csv"))
    all_csv = csv_preprocess(all_csv)
    if args.split_option == "different":        
        if args.stratify:
            train_csv, val_csv = train_test_split(
                all_csv,
                test_size=args.val_ratio,
                shuffle=True,
                random_state=args.seed,
                stratify=all_csv["age_category"],  # age_class stratify
            )
        else:
            train_csv, val_csv = train_test_split(
                all_csv,
                test_size=args.val_ratio,
                shuffle=True,
                random_state=args.seed,  # no stratify
            )
        train_data = increment_path(os.path.join(args.train_dir, "images"), train_csv)
        val_data = increment_path(os.path.join(args.train_dir, "images"), val_csv)
    else:
        data = increment_path(os.path.join(args.train_dir, "images"), all_csv)
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

    ### age preprocess ###
    if args.age_pred == 'regression':
        if args.age_normalized == 'normal':
            Statistic = namedtuple('Statistic', ['mean', 'std'])
            tmp = train_data[:,2].astype(np.float)
            age_stat = Statistic(mean=tmp.mean(),std=tmp.std())
            train_data[:,2] = (tmp-age_stat.mean)/age_stat.std

            tmp = val_data[:,2].astype(np.float)
            val_data[:,2] = (tmp-age_stat.mean)/age_stat.std
        elif args.age_normalized == 'minmax':
            Statistic = namedtuple('Statistic', ['min', 'max'])
            tmp = train_data[:,2].astype(np.float)
            age_stat = Statistic(min=tmp.min(),max=tmp.max())
            train_data[:,2] = (tmp-age_stat.min)/(age_stat.max -age_stat.min)

            tmp = val_data[:,2].astype(np.float)
            val_data[:,2] = (tmp-age_stat.min)/(age_stat.max -age_stat.min)

    train_transform, val_transform = build_transform(
        args=args, phase="train"
    )  # data augmentation
    train_dataset = MulitaskDataset(
        args.train_dir, train_data, transform=train_transform
    )
    val_dataset = MulitaskDataset(args.train_dir, val_data, transform=val_transform)

    if args.wrs:
        print("WeightedRandomSampler")
        # target = [train_dataset[i][4] for i in range(len(train_dataset))] # target : age category 
        # class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        # weight = 1. / class_sample_count
        # samples_weight = np.array([weight[t] for t in target])

        target = train_data[:,3] # target : age category 
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[int(t)] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.n_workers)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
        )
        
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    model = MultitaskModel(args).cuda()
    loss_fn = MultitaskLoss(args).cuda()

    optimizer = Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # optimizer = SGD(
    #     [param for param in model.parameters() if param.requires_grad],
    #     lr=base_lr, weight_decay=1e-4, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10)


    best_epoch, best_score = 0, 0
    for epoch in range(1, args.num_epochs + 1):
        print("### epoch {} ###".format(epoch))
        ### train ###
        train(
            args=args,
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            age_stat=age_stat,
        )

        ### validation ###
        score = validation(
            args=args,
            epoch=epoch,
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            age_stat=age_stat,
        )

        ### save model ###
        if best_score < score:
            best_epoch, best_score = epoch, score
            # torch.save(
            #     {"model": model.state_dict()},
            #     os.path.join(save_path, "model_" + str(epoch) + ".pth"),
            # )
            torch.save(
                {"model": model.state_dict()},
                os.path.join(save_path, "best_model.pth"),
            )
            print(">>>>>> SAVED model at {:02d}".format(epoch))
        print("[best] epoch: {}, score : {:.4f}\n".format(best_epoch, best_score))
    wandb.finish()
