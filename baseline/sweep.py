import os
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from utils import *
from metric import accuracy, macro_f1
from dataset import CustomDataset
from model import CustomModel
from process import train, validation

"""
TODO
TTA
data preprocess (ex. background subtraction)
"""

def main():
    global global_args,best_score,num_save
    wandb.init()
    wandb.config.update(global_args)
    args = wandb.config
    

    train_csv = pd.read_csv(os.path.join(args.train_dir, "train.csv"))
    data = csv_preprocess(os.path.join(args.train_dir, "images"), train_csv)

    train_data, val_data = train_test_split(
        data, test_size=args.val_ratio, shuffle=True, random_state=args.seed
    )

    train_transform, val_transform = build_transform(args=args, phase="train") # data augmentation

    train_dataset = CustomDataset(args.train_dir, train_data, transform=train_transform) 
    val_dataset = CustomDataset(args.train_dir, val_data, transform=val_transform) 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.n_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.n_workers)

    model = CustomModel(args).cuda()

    optimizer = Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # optimizer = SGD(
    #     [param for param in model.parameters() if param.requires_grad],
    #     lr=base_lr, weight_decay=1e-4, momentum=0.9)

    #scheduler = StepLR(optimizer, step_size=10, gamma=0.1) 
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    loss_fn = nn.CrossEntropyLoss().cuda()

    best_score = 0
    for epoch in range(1,args.num_epochs+1):
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
        )

        ### validation ###
        score = validation(
            args=args,
            epoch=epoch,
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
        )

        ### save model ###
        if best_score < score:
            best_score = score
            torch.save(
                {"model": model.state_dict()},
                os.path.join(save_path, "model_" + str(num_save) + ".pth"),
            )
            num_save+=1
            print(">> SAVED model at {:02d}".format(epoch))
            print("lr : {:03f} batch size : {:d} num epoch : {d}".format(args.lr,args.batch_size,args.num_epochs))
        #print("max model : {}, max score : {:.4f}\n".format(, best_score))
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.3) # train-val slit ratio
    #parser.add_argument("--num_epochs", type=int, default=50) 

    #parser.add_argument("--lr", type=float, default=0.01) 
    parser.add_argument("--weight_decay", type=float, default=1e-4) 
    #parser.add_argument("--batch_size", type=int, default=64) 

    # parser.add_argument("--in_size", type=int, default=224) # input size image
    parser.add_argument("--n_workers", type=int, default=4) 

    # parser.add_argument("--print_iter", type=int, default=10)
    # parser.add_argument("--num_classes", type=int, default=100)
    # parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument("--train_dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--save_dir", type=str, default="/opt/ml/experiment/")
    parser.add_argument("--project_name", type=str, default="baseline")
    parser.add_argument("--experiment_name", type=str, default="sweep_test")
    parser.add_argument("--backbone_name", type=str, default="resnet50")
    global_args = parser.parse_args()

    save_path = os.path.join(global_args.save_dir,global_args.project_name,global_args.experiment_name)
    os.makedirs(save_path, exist_ok=False)
    
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize', 
            'name': 'val_loss'
            },
        'parameters': {
            'batch_size': {'values': [32, 64, 128]},
            'num_epochs': {'values': [5, 10, 15]},
            'lr': {'max': 0.1, 'min': 0.0001}
        }
    }
    
    best_score,num_save = 0,0
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')
    wandb.agent(sweep_id, function=main, count=4) # count : 실행 횟수