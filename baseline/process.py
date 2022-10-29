import os
from datetime import datetime
from collections import defaultdict

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
    # info = {"total loss":0,"gen loss":0,"age loss":0,"mask loss":0}
    info, time = defaultdict(int), datetime.now()
    model.train()
    for img, label, gen, age, age_category, mask in loader:
        img, label, gen, age, age_category, mask = (
            img.cuda(),
            label.cuda(),
            gen.cuda(),
            age.cuda(),
            age_category.cuda(),
            mask.cuda(),
        )
        gen_pred, age_pred, mask_pred = model(img)

        gen_pred, age_pred, mask_pred = model(img)
        gen_loss, age_loss, mask_loss = loss_fn(
            gen_pred, age_pred, mask_pred, gen, age_category, mask
        )
        loss = gen_loss + age_loss + mask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = make_class(gen_pred.cpu(), age_pred.cpu(), mask_pred.cpu())
        preds = torch.cat((preds, pred))
        labels = torch.cat((labels, label.cpu()))
        info["train_total_loss"] += loss.item() / len(loader)
        info["train_gen_loss"] += gen_loss.item() / len(loader)
        info["train_age_loss"] += age_loss.item() / len(loader)
        info["train_mask_loss"] += mask_loss.item() / len(loader)

    scheduler.step()
    info["train_f1"] = macro_f1(labels, preds)
    info["train_acc"] = accuracy(labels, preds)
    info["train_elapsed"] = str(datetime.now() - time)
    info["train_epoch"] = epoch

    print("[train]", info)
    wandb.log(info)


def validation(args, epoch, model, loader, loss_fn):
    model.eval()
    preds, labels = torch.tensor([]), torch.tensor([])
    info, time = defaultdict(int), datetime.now()
    with torch.no_grad():
        # for img, label in tqdm(loader):
        for img, label, gen, age, age_category, mask in loader:
            img, label, gen, age, age_category, mask = (
                img.cuda(),
                label.cuda(),
                gen.cuda(),
                age.cuda(),
                age_category.cuda(),
                mask.cuda(),
            )
            gen_pred, age_pred, mask_pred = model(img)

            gen_pred, age_pred, mask_pred = model(img)
            gen_loss, age_loss, mask_loss = loss_fn(
                gen_pred, age_pred, mask_pred, gen, age_category, mask
            )
            loss = gen_loss + age_loss + mask_loss

            pred = make_class(gen_pred.cpu(), age_pred.cpu(), mask_pred.cpu())
            preds = torch.cat((preds, pred))
            labels = torch.cat((labels, label.cpu()))
            info["val_total_loss"] += loss.item() / len(loader)

    info["val_f1"] = macro_f1(labels, preds)
    info["val_acc"] = accuracy(labels, preds)
    info["val_elapsed"] =  str(datetime.now() - time)
    info["epoch"] = epoch

    print("[validation]", info)
    wandb.log(info)
    return info["f1"]
