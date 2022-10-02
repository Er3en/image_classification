from torchvision import transforms
import torch
def transform(arg,cfg):
    print(arg)
    data_transform = {
    'train': transforms.Compose([transforms.Resize([224,224]),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.Normalize(mean=torch.tensor([0.5,0.4,0.3])
                                 ,std=torch.tensor([0.2,0.2,0.2])),
                                 transforms.ToTensor()]),

    'val':transforms.Compose([transforms.Resize(cfg.dataset.img_size),
                              transforms.Resize(224),
                              transforms.ToTensor()])
    }
    return data_transform[arg]

