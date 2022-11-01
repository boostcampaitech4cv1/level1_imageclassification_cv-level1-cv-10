import os
import argparse
import random

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from dataset import TestDataset
from model import CustomModel, MultitaskModel

"""
TODO
git setting
loss viz (tensorboard or wandb)
write log
hyperparameter tunning tool (ray tune or optuna or )
TTA
data preprocess (ex. background subtraction)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--val_ratio", type=float, default=0.3)  # train-val slit ratio
    # parser.add_argument("--split_option", type=str, default="none")  # different or none
    # parser.add_argument("--stratify", type=bool, default=False)
    # parser.add_argument("--wrs", type=bool, default=True)

    parser.add_argument("--age_pred", type=str, default="classification") # classification or regression or ordinary
    parser.add_argument("--age_normalized", type=str, default="normal") # normal or minmax

    parser.add_argument("--in_size", type=int, default=256) # input size image
    parser.add_argument("--crop_type", type=str, default="center") # crop type : center or random or random_resized
    parser.add_argument("--degrees", type=int, default=0) # rotation degree
    parser.add_argument("--translate", type=float, default=0.1) # translate ratio

    parser.add_argument("--num_epochs", type=int, default=30)
    # parser.add_argument("--lr", type=float, default=0.01)
    # parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_workers", type=int, default=4)

    parser.add_argument("--backbone_name", type=str, default="resnet50")
    parser.add_argument("--test_dir", type=str, default="/opt/ml/input/data/eval")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/opt/ml/experiment/augmentation_test/7(11.01 10:50)",
    )
    parser.add_argument("--target_model", type=str, default="best_model.pth")
    args = parser.parse_args()

    submission = pd.read_csv(os.path.join(args.test_dir, "info.csv"))
    test_transform = build_transform(args=args, phase="test")
    test_dataset = TestDataset(
        os.path.join(args.test_dir, "images"),
        submission["ImageID"],
        transform=test_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultitaskModel(args).cuda()

    ### load model ###
    ckpt = torch.load(os.path.join(args.save_dir, args.target_model))
    model.load_state_dict(ckpt["model"])
    print("LOADED model")
    ### test ###
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for img in tqdm(test_loader):
            img = img.cuda()
            gen_pred, age_pred, mask_pred = model(img)
            pred = make_class(args, gen_pred, age_pred, mask_pred)
            all_predictions.extend(pred.cpu().numpy())
    submission["ans"] = all_predictions
    submission.to_csv(
        os.path.join(args.save_dir, args.target_model.split(".")[0] + ".csv"),
        index=False,
    )
    print("test inference is done!")
