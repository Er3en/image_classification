from torch import nn
from collections import OrderedDict
from torchvision import models
import timm
class Model(nn.Module):
    def __init__(self, model_name, labels, pretrained=True):    
        num_labels = len(labels)
        self.pretrained = pretrained
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Softmax(dim=1)
        self.model = models.vgg19(pretrained=self.pretrained)
        self.model.classifier =  nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 6000)),
                                         ('relu', nn.ReLU()),
                                         ('dropout', nn.Dropout(.5)), 
                                         ('fc2', nn.Linear(6000, 10)), 
                                         ('output', nn.Softmax(dim=1) )])) 
                                      
    def forward(self, images):
        features = self.model(images)
        features = self.dropout(features)
        output = self.fc(features)
        return output
    


class A_Model(nn.Module):
    def __init__(self, model_name, labels, pretrained=True):
        super(A_Model, self).__init__()
        
        num_labels = len(labels)
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_labels)
        self.fc = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, images):
        features = self.model(images)
        features = self.dropout(features)
        output = self.fc(features)
        return output
    
