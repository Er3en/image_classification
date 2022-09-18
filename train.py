import torch
import hydra
import numpy as np
from src.dataloader import *

@hydra.main(config_path="cfg", config_name="config")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: device {device}")
    print(cfg.path)
    train_dataset = ClassificationDataset(cfg.path)
    

if __name__ == "__main__":
    main()