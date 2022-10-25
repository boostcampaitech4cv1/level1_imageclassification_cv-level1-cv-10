import os
from datetime import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from metric import accuracy,macro_f1
from dataset import CustomDataset
from model import CustomModel


def train(args, model, loader, optimizer, scheduler,loss_fn,data_len):
    preds, labels = torch.tensor([]), torch.tensor([])
    total_loss,time = 0,datetime.now()
    for img, label in tqdm(loader):
        img,label = img.cuda(),label.cuda()
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
    f1 = macro_f1(labels,preds)
    acc = accuracy(labels,preds)
    elapsed = (datetime.now() - time)
    print('[train] loss {:.3f} | f1 {:.3f} | acc {:.3f} | elapsed {}'.format(total_loss/data_len,f1,acc,elapsed))

def validation(args,model,loader,loss_fn,data_len):
    model.eval()
    preds, labels = torch.tensor([]), torch.tensor([])
    total_loss,time = 0,datetime.now()
    with torch.no_grad():
        for img, label in tqdm(loader):
            img,label = img.cuda(),label.cuda()
            logit = model(img)
            loss = loss_fn(logit, label)
            pred = logit.argmax(dim=1)

            preds = torch.cat((preds, pred.cpu()))
            labels = torch.cat((labels, label.cpu()))
            total_loss += loss.item()

    f1 = macro_f1(labels,preds)
    acc = accuracy(labels,preds)
    elapsed = (datetime.now() - time)
    print('[validation] loss {:.3f} | f1 {:.3f} | acc {:.3f} | elapsed {}'.format(total_loss/data_len,f1,acc,elapsed))
    return f1