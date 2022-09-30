from torchvision import transforms
def transform_train(cfg):
    data_transform = transforms.Compose([transforms.Resize(cfg.dataset.img_size),
                                         transforms.Resize(224),
                                         transforms.ToTensor()
                                        ])
    return data_transform
def transform_val(cfg):
    data_transform = transforms.Compose([transforms.Resize(cfg.dataset.img_size),
                                         transforms.Resize(224),
                                         transforms.ToTensor()
                                        ])
    return data_transform