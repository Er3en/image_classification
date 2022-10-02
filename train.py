from collections import defaultdict
import time
import torch
import gc
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import create_dataloaders
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch import nn
from model import A_Model
from tqdm import tqdm
import wandb

def criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs,targets)

def train_one_ep(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    bar = tqdm(enumerate(dataloader),total=len(dataloader))
    
    for step, data in  bar:
        dataset_size = 0
        running_loss = 0.0
       
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.long)
        targets = torch.reshape(targets, (-1,.0))
        batch_size = images.size(0)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
       
        if scheduler is not None:
            scheduler.stop()
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss= epoch_loss, LR=optimizer.param_groups[0]['lr'])
        gc.collect()
        return epoch_loss



@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: device {device}") 
    train_loader, valid_loader, labels = create_dataloaders()
    #train_loader.__getitem__(2)
    
    model = A_Model('vgg',labels=labels,pretrained=True)  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), cfg.trainer.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=1000)
    start = time.time()
    best_epoch_loss = np.inf
    history = defaultdict(list)
    for epoch in range(cfg.trainer.epochs):
        gc.collect()
        train_epoch_loss = train_one_ep(model, optimizer, scheduler,dataloader=train_loader,device=device, epoch=epoch)
        val_epoch_loss = train_one_ep(model, optimizer, scheduler,dataloader=train_loader,device=device, epoch=epoch)
        wandb.log({'Train loss': train_epoch_loss})
    end = time.time()
    time_elapsed = end - start

if __name__ == "__main__":
    main()