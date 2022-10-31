from email.mime import image
import os

from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import csv_preprocess

class ValancedDataset(Dataset):  # for train and validation
    def __init__(self, root, data, male_30_transform, 
                                    female_30_transform,
                                    male_3060_transform,
                                    female_3060_transform,
                                    male_60_transform,
                                    female_60_transform):

        self.root = os.path.join(root, "images")
        self.data = data
        self.male_30_transform = male_30_transform
        self.female_30_transform = female_30_transform
        self.male_3060_transform = male_3060_transform
        self.female_3060_transform = female_3060_transform
        self.male_60_transform = male_60_transform
        self.female_60_transform = female_60_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, gen, age, age_category, mask, label, img_path = self.data[idx]
        ### 실제 데이터 다 넘김 ###
        gen, age, age_category, mask, label = (
            int(gen),
            int(age),
            int(age_category),
            int(mask),
            int(label),
        )
        path = os.path.join(self.root, img_path)
        image = Image.open(path)
        if (gen == 0) and (age_category == 0):
            # pass
            transform_seq = self.male_30_transform
            len_transform = len(transform_seq)
            for i in range(len_transform):
                transform = transform_seq[i]
                image0 = transform(image)
                file_name,_ = path.split('.')
                # image0.save(f'/opt/ml/input/data/train/images/aug/{i}.jpg')
                image0.save(''.join([file_name,f'_aug{i}.jpg']))

        elif (gen == 1) and (age_category == 0):
            # pass
            transform_seq = self.female_30_transform
            len_transform = len(transform_seq)
            for j in range(len_transform):
                transform = transform_seq[j]
                image0 = transform(image)
                file_name,_ = path.split('.')
                # image0.save(f'/opt/ml/input/data/train/images/aug/{j}.jpg')
                image0.save(''.join([file_name,f'_aug{j}.jpg']))

        elif (gen == 0) and (age_category == 1):
            # pass
            transform_seq = self.male_3060_transform
            len_transform = len(transform_seq)
            for q in range(len_transform):
                transform = transform_seq[q]
                image0 = transform(image)
                file_name,_ = path.split('.')
                # image0.save(f'/opt/ml/input/data/train/images/aug/{q}.jpg')
                image0.save(''.join([file_name,f'_aug{q}.jpg']))

        elif (gen == 1) and (age_category == 1):
            # pass
            transform_seq = self.female_3060_transform
            len_transform = len(transform_seq)
            for t in range(len_transform):
                transform = transform_seq[t]
                image0 = transform(image)
                file_name,_ = path.split('.')
                # image0.save(f'/opt/ml/input/data/train/images/aug/{t}.jpg')
                image0.save(''.join([file_name,f'_aug{t}.jpg']))

        elif (gen == 0) and (age_category == 2):
            # pass
            transform_seq = self.male_60_transform
            len_transform = len(transform_seq)
            for l in range(len_transform):
                transform = transform_seq[l]
                image0 = transform(image)
                file_name,_ = path.split('.')
                # image0.save(f'/opt/ml/input/data/train/images/aug/{l}.jpg')
                image0.save(''.join([file_name,f'_aug{l}.jpg']))      

        elif (gen == 1) and (age_category == 2):
            # pass
            transform_seq = self.female_60_transform
            len_transform = len(transform_seq)        
            for m in range(len_transform):
                transform = transform_seq[m]
                image0 = transform(image)
                file_name,_ = path.split('.')
                # image0.save(f'/opt/ml/input/data/train/images/aug/{m}.jpg')
                image0.save(''.join([file_name,f'_aug{m}.jpg']))  

        return 'New Valance!'

if __name__ == "__main__":
    train_dir = "/opt/ml/input/data/train"

    male_30_transform = torch.nn.Sequential(         # 2 augmentations -> 3배

            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            # transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3)

    )
    female_30_transform = torch.nn.Sequential(       # 1 augmentations -> 2배
        
            transforms.RandomHorizontalFlip(p=1.0),
            
        
    )
    male_3060_transform = torch.nn.Sequential(      # 3 augmentations -> 4배
        
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            
        
    )
    female_3060_transform = torch.nn.Sequential(    # 1 augmentations -> 2배
        
            transforms.RandomHorizontalFlip(p=1.0),
              
        
    )
    male_60_transform = torch.nn.Sequential(        # 9 augmentaions -> 10배
        
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
               
        
    )
    female_60_transform = torch.nn.Sequential(      # 9 augmentaions -> 10배
        
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),
            transforms.ColorJitter(brightness= .5,contrast=0.5,saturation=0.5,hue=0.3),

          
    )
    train_csv = pd.read_csv(os.path.join(train_dir, "train.csv"))
    data = csv_preprocess(os.path.join(train_dir, "images"), train_csv)

    valanced_data = ValancedDataset(train_dir, data, male_30_transform,
                                                            female_30_transform,
                                                            male_3060_transform,
                                                            female_3060_transform,
                                                            male_60_transform,
                                                            female_60_transform)

    train_loader = DataLoader(valanced_data, batch_size=1, shuffle=False)

    for label in tqdm(train_loader):
        pass