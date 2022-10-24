import os
import numpy as np

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