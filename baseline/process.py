import os
from datetime import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb

import torch
import torch.nn as nn

from utils import *
from metric import accuracy, macro_f1

def train(args, epoch, model, loader, optimizer, scheduler, loss_fn):
    preds, labels = torch.tensor([]), torch.tensor([])
    total_loss, time = 0, datetime.now()
    #for img, label in tqdm(loader):
    for img, label in loader:
        img, label = img.cuda(), label.cuda()
        logit = model(img)
        loss = loss_fn(logit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logit.argmax(dim=1)
        preds = torch.cat((preds, pred.cpu()))
        labels = torch.cat((labels, label.cpu()))
        total_loss += loss.item()
    scheduler.step()
    f1 = macro_f1(labels, preds)
    acc = accuracy(labels, preds)
    elapsed = datetime.now() - time
    avg_loss = total_loss/len(loader)
    print(
        "[train] loss {:.3f} | f1 {:.3f} | acc {:.3f} | elapsed {}".format(
            total_loss / len(loader), f1, acc, elapsed
        )
    )
    wandb.log({"epoch":epoch,"train_loss":avg_loss,"train_f1":f1,"train_acc":acc})


def validation(args, epoch, model, loader, loss_fn):
    model.eval()
    preds, labels = torch.tensor([]), torch.tensor([])
    total_loss, time = 0, datetime.now()
    with torch.no_grad():
        #for img, label in tqdm(loader):
        for img, label in loader:
            img, label = img.cuda(), label.cuda()
            logit = model(img)
            loss = loss_fn(logit, label)
            pred = logit.argmax(dim=1)

            preds = torch.cat((preds, pred.cpu()))
            labels = torch.cat((labels, label.cpu()))
            total_loss += loss.item()

    f1 = macro_f1(labels, preds)
    acc = accuracy(labels, preds)
    elapsed = datetime.now() - time
    avg_loss = total_loss/len(loader)
    print(
        "[validation] loss {:.3f} | f1 {:.3f} | acc {:.3f} | elapsed {}".format(
            avg_loss, f1, acc, elapsed
        )
    )
    wandb.log({"epoch":epoch,"val_loss":avg_loss,"val_f1":f1,"val_acc":acc})
    return f1
