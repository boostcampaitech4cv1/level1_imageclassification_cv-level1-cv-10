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


def train(args, epoch, model, loader, optimizer, scheduler, loss_fn, age_stat=None):
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

        if args.age_pred == 'classification':
            gen_loss, age_loss, mask_loss = loss_fn(
                gen_pred, age_pred, mask_pred, gen, age_category, mask
            )
        elif args.age_pred == 'regression':
            age = age.unsqueeze(-1)
            gen_loss, age_loss, mask_loss = loss_fn(
                gen_pred, age_pred, mask_pred, gen, age, mask
            )
        elif args.age_pred == 'ordinary':
            pass
        elif args.age_pred == 'cls_regression':
            age_category = age_category.float().unsqueeze(-1)
            gen_loss, age_loss, mask_loss = loss_fn(
                gen_pred, age_pred, mask_pred, gen, age_category, mask
            )
        loss = gen_loss + age_loss + mask_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = make_class(args, gen_pred.cpu(), age_pred.cpu(), mask_pred.cpu(), age_stat)
        preds = torch.cat((preds, pred))
        labels = torch.cat((labels, label.cpu()))
        info["train_total_loss"] += loss.item() / len(loader)
        info["train_gen_loss"] += gen_loss.item() / len(loader)
        info["train_age_loss"] += age_loss.item() / len(loader)
        info["train_mask_loss"] += mask_loss.item() / len(loader)

    scheduler.step()

    info["epoch"] = epoch
    info["train_f1"] = macro_f1(labels, preds)
    info["train_acc"] = accuracy(labels, preds)
    elapsed = datetime.now() - time

    print(
        "[train] f1 {:.3f} | acc {:.3f} | elapsed {}".format(
            info["train_f1"], info["train_acc"], elapsed
        )
    )
    print(
        "[train] total loss {:.3f} | gen loss {:.3f} | age loss {:.3f} | mask loss {:.3f}".format(
            info["train_total_loss"],
            info["train_gen_loss"],
            info["train_age_loss"],
            info["train_mask_loss"],
        )
    )
    wandb.log(info)


def validation(args, epoch, model, loader, loss_fn, age_stat=None):
    model.eval()
    preds, labels = torch.tensor([]), torch.tensor([])
    info, time, num = defaultdict(int), datetime.now(), 0
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
            if args.age_pred == 'classification':
                gen_loss, age_loss, mask_loss = loss_fn(
                    gen_pred, age_pred, mask_pred, gen, age_category, mask
                )
            elif args.age_pred == 'regression':
                age = age.unsqueeze(-1)
                gen_loss, age_loss, mask_loss = loss_fn(
                    gen_pred, age_pred, mask_pred, gen, age, mask
                )
            elif args.age_pred == 'ordinary':
                pass
            elif args.age_pred == 'cls_regression':
                age_category = age_category.float().unsqueeze(-1)
                gen_loss, age_loss, mask_loss = loss_fn(
                    gen_pred, age_pred, mask_pred, gen, age_category, mask
                )
            loss = gen_loss + age_loss + mask_loss
            pred = make_class(args, gen_pred.cpu(), age_pred.cpu(), mask_pred.cpu(), age_stat)
            preds = torch.cat((preds, pred))
            labels = torch.cat((labels, label.cpu()))
            info["val_total_loss"] += loss.item() / len(loader)

    f1_all = macro_f1(labels, preds, return_all=True)
    info["val_f1"] = sum(f1_all) / len(f1_all)
    info["val_gen_f1"] = macro_f1((labels // 3) % 2, (preds // 3) % 2)
    age_f1_all = macro_f1(labels % 3, preds % 3, return_all=True)
    info["val_age_f1"] = sum(age_f1_all) / len(age_f1_all)
    info["val_age_f1_0"], info["val_age_f1_1"], info["val_age_f1_2"] = age_f1_all

    info["val_mask_f1"] = macro_f1(labels // 6, preds // 6)

    info["epoch"] = epoch
    info["val_acc"] = accuracy(labels, preds)
    elapsed = datetime.now() - time

    print(
        "[val] f1 {:.3f} | acc {:.3f} | loss {:.3f} | elapsed {}".format(
            info["val_f1"], info["val_acc"], info["val_total_loss"], elapsed
        )
    )
    print(
        "[val] gen f1 {:.3f} | age f1 {:.3f} | mask f1 {:.3f} ".format(
            info["val_gen_f1"], info["val_age_f1"], info["val_mask_f1"]
        )
    )
    print(
        "[val] age=0 f1 {:.3f} | age=1 f1 {:.3f} | age=2 f1 {:.3f} ".format(
            info["val_age_f1_0"], info["val_age_f1_1"], info["val_age_f1_2"]
        )
    )
    print(
        "[val] 5 smallest f1",
        dict(
            sorted(
                [(i, round(f1_all[i], 3)) for i in range(len(f1_all))],
                key=lambda x: x[1],
            )[:5]
        ),
    )
    wandb.log(info)
    return info["val_f1"]
