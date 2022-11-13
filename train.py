from collections import defaultdict
import time
import torch
import gc
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import lr_scheduler
from utils import create_dataloaders
from utils import train_step, validation_step
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch import nn
from model import A_Model
from tqdm import tqdm
import wandb
import omegaconf
from copy import deepcopy
import copy


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: device {device}") 
    train_loader, valid_loader, labels = create_dataloaders()
    
    
    model = A_Model('tf_efficientnet_b0',labels=labels,pretrained=True)
    optimizer = optim.SGD(model.parameters(), cfg.trainer.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=1000, eta_min=cfg.trainer.lr)
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    for epoch in range(cfg.trainer.epochs):
        gc.collect()
        train_epoch_loss = train_step(model, optimizer, scheduler, train_loader, device, epoch)
        val_epoch_loss = validation_step(model, optimizer, scheduler, valid_loader, device, epoch)
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
 
        if val_epoch_loss <= best_epoch_loss:
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),'logs/Loss{:.4f}_epoch{:.0f}.bin'.format(best_epoch_loss, epoch))
        print(history)
    end = time.time()
    time_elapsed = end - start
    print(f"Lenght of training {time_elapsed}")

if __name__ == "__main__":
    main()