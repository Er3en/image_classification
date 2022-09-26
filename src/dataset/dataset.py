import os
import torch
import pandas as pd
from pathlib import Path
from src.utils import *
from typing import Tuple


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                data_path: Path, 
                batch_size: int = 16, 
                shuffle: bool = True,
                tile_size: Tuple = (256,256),
                random_state: int = 16,
                show_image: bool = False,
                ):
       
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tile_size = tile_size
        self.random_state = random_state
        self.labels = {'cane': 'dog',
          'cavallo': 'horse',
          'elefante': 'elephant',
          'farfalla': 'butterfly',
          'gallina': 'chicken',
          'gatto': 'cat',
          'mucca': 'cow',
          'pecora': 'sheep',
          'ragno': 'spider',
          'scoiattolo': 'squirrel',
         }
    
        self_dataframe = self._create_df()


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        item = self.dataset[idx]
        return item

    def _create_df(self):
        data = {'path': [],'label': []}
        for  dir in os.listdir(self.data_path):
            filename = os.listdir(f'{self.data_path}/{dir}')    
            for file in filename:
                data['path'].append(f'{self.data_path}/{dir}/{file}')
                data['label'].append(self.labels[dir])
        df_data = pd.DataFrame(data)
        return df_data
       
