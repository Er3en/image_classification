from cProfile import label
import torch
import hydra
import numpy as np
import torch.optim as optim
from utils import create_dataloaders
import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn
from model import Model

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: device {device}") 
    train_loader, valid_loader, labels = create_dataloaders()
    
    
    model = Model('vgg',labels=labels,pretrained=True)  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), cfg.trainer.lr)

    

if __name__ == "__main__":
    main()