import os
import torch
import pandas as pd
import numpy as np
import cv2 
from pathlib import Path
from src.utils import *
from typing import Tuple
from sklearn.utils import shuffle

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                data_path:Path, 
                batch_size:int = 16, 
                shuffle: bool = True,
                tile_size: Tuple = (256,256),
                random_state: int = 16,
                show_image: bool = True
                ):
        self.dict = {'scoiattolo':'squirrel','cavallo':'horse',
            'farfalla':'butterfly','mucca':'cow','gatto':'cat', 
            'pecora':'sheep','gallina':'chicken','cane':'dog',
            'elefante':'elephant','ragno':'spider'
            }
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tile_size = tile_size
        self.random_state = random_state
        self.dataset = self._get_classification_ds()
        self.dataframe = self.dataset
        self.dataset = self._read_images()
        self.show_image = show_image
        if self.show_image: 
            show_images(self.dataframe, self.dataset, 2, 5)
    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        item = self.dataset[idx]
        return item

     

    def _get_classification_ds(self):
        folders = os.listdir(self.data_path)
        categories = []
        files = []
        for dir in folders:
            filename = os.listdir(self.data_path + dir)
            if dir in self.dict.keys():
                cat = self.dict[dir]
            for file in filename:
                files.append(self.data_path+dir+"/"+file)
                categories.append(cat)

        df = pd.DataFrame({"filename": files, "category": categories})
        train_df = pd.DataFrame(columns=['filename', 'category'])
        for i in range(len(folders)):
            train_df = train_df.append(df[df.category == i].iloc[:500,:])

        train_df.head()
        train_df = train_df.reset_index(drop=True)
        if self.shuffle:
            y = train_df['category']
            x = train_df['filename']
            x, y = shuffle(x, y,random_state=self.random_state)
        return train_df

    def _read_images(self):
        images = []
        bar = progress_bar(self.dataset)
        for idx, file_path in enumerate(self.dataset.filename.values):
            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            img = center_image(cv2.resize(img, self.tile_size))
            images+=[img]
            bar.update(idx + 1)
        images = np.array(images)
        return images
