import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

import torch
import gc
from tqdm import tqdm
from PIL import Image
from progressbar import *
from sklearn.model_selection import train_test_split
from hydra import compose, initialize
from omegaconf import OmegaConf
from rich.progress import track
from torch.utils.data import DataLoader
from torch import nn

from dataset import Classification_Dataset
from transforms import transform

def center_image(img):
    size = [256, 256]
    img_size = img.shape[:2]
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img
    return resized

def progress_bar(array):
    widgets = [' Progress: ', SimpleProgress(),
               ', Percent: ', Percentage(),
               ' ', ETA(),
               ' ', AdaptiveETA()]
    
    bar = progressbar.ProgressBar(maxval=len(array),widgets=widgets).start()
    return bar


def show_images(df):
    plt.figure(figsize = (20, 6))
    for idx, i in enumerate(df.label.unique()):
        plt.subplot(1, 10, idx + 1)
        df = df[df['label'] == i].reset_index(drop=True)
        image_path = df.loc[random.randint(0, len(df) - 1), 'path']
        img = Image.open(image_path)
        img = img.resize((224,224))
        plt.imshow(img)
        plt.axis('off')
        plt.title(i)

    plt.tight_layout()
    plt.show()


def get_labels():
    labels = {
    'cane': 'dog',
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
    return labels

def create_df():
    cfg = OmegaConf.load("cfg/config.yaml")
    cfg_ds = cfg.dataset
    data = {'path': [],'label': []}
    labels = get_labels()

    for  dir in os.listdir(cfg_ds.path):
        filename = os.listdir(f'{cfg_ds.path}/{dir}')    
        for file in filename:
            data['path'].append(f'{cfg_ds.path}/{dir}/{file}')
            data['label'].append(labels[dir])
    df_data = pd.DataFrame(data)
    return df_data, labels


def create_dataloaders():
    cfg = OmegaConf.load("cfg/config.yaml")

    cfg_ds = cfg.dataset
    df, labels = create_df()
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=cfg.dataset.random_state, stratify=df['label'])
    
    train_dataset = Classification_Dataset(
                                           dataframe=df_train,
                                           labels=labels, 
                                           shuffle=True, 
                                           trans=transform('train', cfg))
    val_dataset = Classification_Dataset(
                                         dataframe=df_valid, 
                                         labels=labels, 
                                         shuffle=True, 
                                         trans=transform('val', cfg))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg_ds.batch_size, 
        shuffle=cfg_ds.shuffle, 
        num_workers=cfg_ds.num_workers,
        pin_memory=cfg_ds.pin_mem
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg_ds.batch_size,                       
        shuffle=cfg_ds.shuffle, 
        num_workers=cfg_ds.num_workers,
        pin_memory=cfg_ds.pin_mem,
        )

    return train_loader, val_loader, labels


def criterion(outputs, label):
    return nn.CrossEntropyLoss()(outputs, label)

def get_bar(dataloader):
   bar = tqdm(enumerate(dataloader), total=len(dataloader))
   return bar


def train_step(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    bar = get_bar(dataloader)
    
    for step, data in  bar:
        dataset_size = 0
        running_loss = 0.0
       
        images = data['image'].to(device, dtype=torch.float)
        label = data['label'].to(device, dtype=torch.long)
        label = torch.reshape(label, (-1,))
   
        batch_size = images.size(0)
        outputs = model(images)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
       
        if scheduler is not None: 
            scheduler.step()
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss= epoch_loss, LR=optimizer.param_groups[0]['lr'])
        gc.collect()
        return epoch_loss

def validation_step(model, optimizer, scheduler, dataloader, device, epoch):
    bar = get_bar(dataloader)

    model.eval()
    dataset_size = 0
    running_loss = 0.0
    for step, data in  bar:
        images = data['image'].to(device, dtype=torch.float)
        label = data['label'].to(device, dtype=torch.long)
        label = torch.reshape(label, (-1,))
        batch_size = images.size(0)

    outputs = model(images)
    loss = criterion(outputs, label)

    running_loss += (loss.item() * batch_size)
    dataset_size += batch_size
    
    epoch_loss = running_loss / dataset_size
    
    bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                    LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    
    return epoch_loss
