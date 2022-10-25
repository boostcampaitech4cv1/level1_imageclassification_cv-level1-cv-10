import os
import torch
import random
import numpy as np

import logging

from torchvision import transforms

'''
### class encoding ###
[Mask] 0 : wear, 1 : incorrect, 2 : not wear
[Gender] 0 : male, 1 : female
[Age] 0 : <30, 1 : >=30 and <60, 3 : >=60
'''
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def csv_preprocess(root,csv_file):
    data = []
    csv_file['gender'] = (csv_file['gender'] == 'female').astype('int')
    csv_file['age_category'] = csv_file['age'].apply(lambda x : 0 if x<30 else (1 if x<60 else 2))
    for _,id,gen,_,age,path,age_category in csv_file.itertuples():
        for img_name in os.listdir(os.path.join(root,path)):
            if img_name[0] !='.':
                if 'normal' in img_name:
                    mask = 2
                elif 'incorrect' in img_name:
                    mask = 1
                else:
                    mask = 0
                label = 6*mask+3*gen+age_category
                data.append((id,gen,age,age_category,mask,label,os.path.join(path,img_name)))
    return np.array(data)

def build_transform(args=None,phase='train'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == 'train':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(degrees=(10,10)),
                #transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        
        val_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(degrees=(10,10)),
                #transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        return train_transform, val_transform
    else:
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        return test_transform

def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        #datefmt='%Y%m%d-%H:%M:%S',
                        datefmt='%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging