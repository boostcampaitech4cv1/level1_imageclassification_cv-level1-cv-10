import os
import argparse
import random
from datetime import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from utils import *
from metric import accuracy,macro_f1
from dataset import CustomDataset
from model import CustomModel

'''
TODO
git setting
loss viz (tensorboard or wandb)
write log
hyperparameter tunning tool (ray tune or optuna or )
TTA
data preprocess (ex. background subtraction)
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=50)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument('--in_size', type=int, default=224)
    parser.add_argument('--n_workers', type=int, default=4)

    #parser.add_argument("--print_iter", type=int, default=10)
    #parser.add_argument("--num_classes", type=int, default=100)
    #parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--train_dir', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--save_dir', type=str, default='/opt/ml/experiment/')
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--backbone_name', type=str, default='resnet50')
    args = parser.parse_args()

    set_seed(args.seed)

    save_path = os.path.join(args.save_dir,args.experiment_name)
    os.makedirs(save_path,exist_ok=False)

    train_csv = pd.read_csv(os.path.join(args.train_dir, 'train.csv'))
    data = csv_preprocess(os.path.join(args.train_dir,'images'),train_csv)
    
    train_data,val_data = train_test_split(data,test_size=args.val_ratio, shuffle=True, random_state=args.seed)
    train_transform, val_transform = build_transform(args=args,phase='train')

    train_dataset = CustomDataset(args.train_dir, train_data, transform=train_transform)
    val_dataset = CustomDataset(args.train_dir, val_data, transform=val_transform)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False)

    model = CustomModel(args).cuda()

    optimizer = Adam([param for param in model.parameters() if param.requires_grad],lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = SGD(
    #     [param for param in model.parameters() if param.requires_grad],
    #     lr=base_lr, weight_decay=1e-4, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    #scheduler = CosineAnnealingLR(optimizer, T_max=10)

    loss_fn = nn.CrossEntropyLoss().cuda()

    best_epoch, best_score = 0,0
    for epoch in range(args.num_epochs):
        print("### epoch {} ###".format(epoch+1))
        ### train ###
        model.train()
        train_preds, train_labels = torch.tensor([]), torch.tensor([])
        train_loss,time = 0,datetime.now()
        for img, label in tqdm(train_loader):
            img,label = img.cuda(),label.cuda()
            logit = model(img)
            loss = loss_fn(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logit.argmax(dim=1)
            train_preds = torch.cat((train_preds, pred.cpu()))
            train_labels = torch.cat((train_labels, label.cpu()))
            train_loss += loss.item()

        scheduler.step()
        train_f1 = macro_f1(train_labels,train_preds)
        train_acc = accuracy(train_labels,train_preds)
        elapsed = (datetime.now() - time)
        print('[train] loss {:.3f} | f1 {:.3f} | acc {:.3f} | elapsed {}'.format(train_loss/len(train_dataset),train_f1,train_acc,elapsed))

        ### validation ###
        model.eval()
        val_preds, val_labels = torch.tensor([]), torch.tensor([])
        val_loss,time = 0,datetime.now()
        with torch.no_grad():
            for img, label in tqdm(train_loader):
                img,label = img.cuda(),label.cuda()
                logit = model(img)
                pred = logit.argmax(dim=1)

                val_preds = torch.cat((val_preds, pred.cpu()))
                val_labels = torch.cat((val_labels, label.cpu()))
                val_loss += loss.item()

        val_f1 = macro_f1(train_labels,train_preds)
        val_acc = accuracy(train_labels,train_preds)
        elapsed = (datetime.now() - time)
        print('[validation] loss {:.3f} | f1 {:.3f} | acc {:.3f} | elapsed {}'.format(val_loss/len(val_dataset),val_f1,val_acc,elapsed))

        ### save model ###
        if best_score<val_f1:
            best_epoch,best_score = epoch, val_f1
            torch.save({'model': model.state_dict()}, os.path.join(save_path, 'model_'+str(epoch)+'.pth'))
            print('>> SAVED model at {:02d}'.format(epoch))
        print('max epoch: {}, max score : {:.4f}'.format(best_epoch, best_score))