from torchvision import transforms
def transform_train(cfg):
    data_transform = transforms.Compose([transforms.Resize(cfg.dataset.size),
                                         transforms.Resize(224),
                                         transforms.ToTensor()
                                        ])

def transform_val(cfg):
    data_transform = transforms.Compose([transforms.Resize(cfg.dataset.size),
                                         transforms.Resize(224),
                                         transforms.ToTensor()
                                        ])