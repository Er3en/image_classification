
import torch
import tqdm as tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
def transform(arg,cfg):
    print(arg)

    data_transform = {
    'train': transforms.Compose([transforms.Resize([cfg.dataset.img_size,cfg.dataset.img_size]),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.Normalize(mean=cfg.dataset.mean,
                                                      std=cfg.dataset.std)
                                ]),

    'val':transforms.Compose([transforms.Resize([cfg.dataset.img_size,cfg.dataset.img_size]),
                              transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)
                            ])
    }
    return data_transform[arg]

def transform_normalize():
    transform = transforms.Compose([transforms.Resize(225), 
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()
                                   ])
    return transform

def calc_std_mean(dataloader):
    mean_sum, std_sum = 0, 0
    batch_size = len(dataloader)
    for data in tqdm.tqdm(dataloader):
        image = data[0].to('cpu',dtype=torch.float32)
        mean_sum += torch.mean(image, dim=[0,2,3])
        std_sum += torch.std(image, dim=[0,2,3])
        
        mean = mean_sum / batch_size
        std = std_sum / batch_size
    return mean, std
#TODO
def normalize():
    data_path = "/home/jarybski/Desktop/Animal-10/archive/raw-img"
    kwargs = {'num_workers':10, 'pin_memory':True, 'persistent_workers':True } if torch.cuda.is_available() else  {}
    dataset = datasets.ImageFolder(data_path ,transform=transform_normalize())
    dataloader = DataLoader(dataset,batch_size=256,shuffle=True,**kwargs)
    mean, std = calc_std_mean(dataloader)
    print(mean,std)
    return mean, std

    
