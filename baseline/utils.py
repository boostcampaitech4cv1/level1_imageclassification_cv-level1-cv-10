import os
import torch
import random
import numpy as np

import logging

from torchvision import transforms

"""
### class encoding ###
[Mask] 0 : wear, 1 : incorrect, 2 : not wear
[Gender] 0 : male, 1 : female
[Age] 0 : <30, 1 : >=30 and <60, 3 : >=60
"""

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def csv_preprocess(csv_file):
    csv_file["gender"] = (csv_file["gender"] == "female").astype("int")
    csv_file["age_category"] = csv_file["age"].apply(
        lambda x: 0 if x < 30 else (1 if x < 60 else 2)
    )
    return csv_file


def increment_path(root, csv_file):
    data = []
    for _, id, gen, _, age, path, age_category in csv_file.itertuples():
        for img_name in os.listdir(os.path.join(root, path)):
            if img_name[0] != ".":
                if "normal" in img_name:
                    mask = 2
                elif "incorrect" in img_name:
                    mask = 1
                else:
                    mask = 0
                label = 6 * mask + 3 * gen + age_category
                data.append(
                    (
                        id,
                        gen,
                        age,
                        age_category,
                        mask,
                        label,
                        os.path.join(path, img_name),
                    )
                )
    return np.array(data)


def build_transform(args=None, phase="train"):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        transform = []

        if args.degrees != 0 or args.translate != 0:
            transform.append(transforms.RandomAffine(degrees=args.degrees,translate=(args.translate,args.translate)))
        if args.crop_type == "center":
            transform.append(transforms.CenterCrop(args.in_size)) # 굳이 centercrop을 안 쓸 이유가 있을까?
        elif args.crop_type == "random":
            transform.append(transforms.RandomCrop(args.in_size))
        elif args.crop_type == "random_resized":
            transform.append(transforms.RandomResizedCrop(args.in_size))

        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=mean, std=std))
        train_transform = transforms.Compose(transform)

        val_transform = transforms.Compose(
            [
                transforms.CenterCrop(args.in_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return train_transform, val_transform
    else:
        ### TTA 추가 가능 ###
        test_transform = transforms.Compose(
            [
                transforms.CenterCrop(args.in_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return test_transform


def make_class(args, gen_pred, age_pred, mask_pred, age_stat=None):
    gen = gen_pred.argmax(dim=1)
    mask = mask_pred.argmax(dim=1)
    if args.age_pred == 'regression':
        age = age_to_class(age_pred.detach(),age_stat,mode=args.age_normalized).squeeze(-1)
    elif args.age_pred == 'classification':
        age = age_pred.argmax(dim=1)
    return 6 * mask + 3 * gen + age


def age_to_class(age,age_stat,mode):
    if mode == 'normal':
        age_class = age*age_stat.std+age_stat.mean
    
    elif mode == 'minmax':
        age_class = age*(age_stat.max-age_stat.min)+age_stat.min
    
    age_class.apply_(
        lambda x: 0 if x < 30 else (1 if x < 60 else 2)
    )
    return age_class