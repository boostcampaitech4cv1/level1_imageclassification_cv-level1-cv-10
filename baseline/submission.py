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
from model import CustomModel

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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--in_size", type=int, default=224)
    parser.add_argument("--n_workers", type=int, default=4)

    # parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--backbone_name", type=str, default="resnet50")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--test_dir", type=str, default="/opt/ml/input/data/eval")
    parser.add_argument("--model_path", type=str, default="/opt/ml/experiment/test")
    args = parser.parse_args()

    submission = pd.read_csv(os.path.join(args.test_dir, "info.csv"))
    test_transform = build_transform(args=args, phase="test")
    test_dataset = TestDataset(
        os.path.join(args.test_dir, "images"),
        submission["ImageID"],
        transform=test_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = CustomModel(args).cuda()

    ### load model ###
    ckpt = torch.load(os.path.join(args.model_path, "model_0.pth"))
    model.load_state_dict(ckpt["model"])
    print("LOADED model")
    ### test ###
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for img in tqdm(test_loader):
            img = img.cuda()
            logit = model(img)
            pred = logit.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission["ans"] = all_predictions
    submission.to_csv(os.path.join(args.model_path, "test.csv"), index=False)
    print("test inference is done!")
