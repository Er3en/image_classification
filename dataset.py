import os
import torch
import torch.utils.data.dataset
from torchvision import transforms
from pandas import DataFrame
from pathlib import Path
from typing import Tuple
import cv2
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets, models, transforms
from PIL import Image
class Classification_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                dataframe=None,
                labels:dict ={},
                transforms=None,
                shuffle: bool = True,
                tile_size: Tuple = (256,256),
                show_image: bool = False,
                ):
              
        self.dataframe = dataframe
        self.labels = labels
        self.transforms = transforms
        self.shuffle = shuffle
        self.tile_size = tile_size
        self.show_image = show_image
        self.label_enc = LabelEncoder()
        self.label_enc.fit(list(self.labels.values()))

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        img_path = item['path']
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print([item['label']])
        
        ###FIX
        if self.transforms:
            transforms = [img.Res]
            img = Image.fromarray(img)

            
            img = self.transforms(img)#['image']
            print(img.shape())

        target = self.label_enc.transform([item['label']])
        print("image shape",img.shape())

        return {'image': img,
                 'target': torch.tensor(target, dtype=torch.long)
        }



