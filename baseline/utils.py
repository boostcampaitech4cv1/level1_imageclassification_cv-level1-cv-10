import os
import numpy as np

from torchvision import transforms
from sklearn.metrics import f1_score

'''
### class encoding ###
[Mask] 0 : wear, 1 : incorrect, 2 : not wear
[Gender] 0 : male, 1 : female
[Age] 0 : <30, 1 : >=30 and <60, 3 : >=60
'''
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
                transforms.RandomResizedCrop(size=256, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(degrees=(10,10)),
                #transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        return test_transform

def custom_f1(label: torch.tensor, predicted: torch.tensor):
    label, predicted = label.tolist(), predicted.tolist()
    label = list(map(re_scoring, label))
    predicted = list(map(logit_to_label, predicted))
    predicted = list(map(re_scoring, predicted))
    #print('<test> label : {:.0f} | predicted : {:.0f}'.format(sum(label),sum(predicted)))
    return f1_score(y_true=label, y_pred=predicted, average='micro')
