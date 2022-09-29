import torch
import hydra
import numpy as np
from datasets import *
from utils import create_dataloaders
import hydra 



@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: device {device}") 

    train_loader, valid_loader = create_dataloaders()
   

if __name__ == "__main__":
    main()