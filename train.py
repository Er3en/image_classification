import torch
import hydra
import numpy as np
from src.dataloader import *
from src.utils import *

@hydra.main(config_path="cfg", config_name="config")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: device {device}")   
    train_dataset = ClassificationDataset(cfg.path)

   

if __name__ == "__main__":
    main()