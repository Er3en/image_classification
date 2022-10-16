from torchvision import transforms
import torch
def transform(arg,cfg):
    print(arg)
    data_transform = {
    'train': transforms.Compose([transforms.Resize([224,224]),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                
                                 ]),

    'val':transforms.Compose([transforms.Resize(cfg.dataset.img_size),
                              transforms.Resize(224),
                              ])
    }
    return data_transform[arg]

