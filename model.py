from torch import nn
from collections import OrderedDict
from torchvision import models

class Model(nn.Module):
    def __init__(self, model_name, labels, pretrained=True):
        super(Model, self).__init__()
        
        self.num_labels = len(labels)
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = models.vgg19(pretrained=self.pretrained)
        self.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088,6000)),
                                                      ('relu',nn.ReLU()),
                                                      ('drop',nn.Dropout(.5)),
                                                      ('fc2',nn.Linear(6000,10)),
                                                      ('output',nn.Softmax(dim=1))
                                                      ]))
        self.model.classifier = self.classifier
     