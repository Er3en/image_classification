from torch import nn
from collections import OrderedDict
from torchvision import models

class A_Model(nn.Module):
    def __init__(self, model_name, labels, pretrained=True):
        super(A_Model, self).__init__()
        
        num_labels = len(labels)
        self.pretrained = pretrained
        self.model = models.vgg19(pretrained=self.pretrained)
        self.fc = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, images):
        features = self.model(images)
        features = self.dropout(features)
        output = self.fc(features)
        return output
    


