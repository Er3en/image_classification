import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path




class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                data_path:Path, 
                batch_size:int = 16, 
                shuffle: bool = True,
                ):
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = self._get_classification_ds()
        if shuffle:
           x = shuffle_array(self.dataset['category'])
           y = shuffle_array(self.dataset['filename'])


    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        return idx

    def _get_classification_ds(self):
        folders = os.listdir(self.data_path)
        categories = []
        files = []
        for k, dir in enumerate(folders):
            filename = os.listdir(self.data_path + dir)
            for file in filename:
                files.append(self.data_path+dir+"/"+file)
                categories.append(k)

        df = pd.DataFrame({"filename": files, "category": categories})
        train_df = pd.DataFrame(columns=['filename', 'category'])
        for i in range(len(folders)):
            train_df = train_df.append(df[df.category == i].iloc[:500,:])

        train_df.head()
        train_df = train_df.reset_index(drop=True)
        return train_df

def shuffle_array(train_df):
    array = np.random.shuffle(train_df)
    return array


