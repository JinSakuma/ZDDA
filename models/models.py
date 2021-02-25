import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .arcface import ArcFace


class AlexNetCNN(nn.Module):
    def __init__(self, base_model):
        super(AlexNetCNN, self).__init__()
        self.base_model = base_model
        self.CNN = nn.Sequential(*list(base_model.children())[:-1])
        
    def forward(self, x):
        output = self.CNN(x)
        return output 
    
    
class AlexNetClassifier(nn.Module):
    def __init__(self, base_model, cls_num=10):
        super(AlexNetClassifier, self).__init__()
        self.cls_num = cls_num
        self.base_model = base_model
        self.classifier = nn.Sequential(*list(self.base_model.children())[-1])
        self.classifier[6] =  nn.Linear(4096, self.cls_num)
        
    def forward(self, x):
        x = x.view(-1, 256*6*6)
        output = self.classifier(x)
        return output 
    
    
class ResNetCNN(nn.Module):
    def __init__(self, base_model):
        super(ResNetCNN, self).__init__()
        self.base_model = base_model
        self.CNN = nn.Sequential(*list(base_model.children())[:-1])
        
    def forward(self, x):
        output = self.CNN(x)
        return output 
    

class ResNetClassifier(nn.Module):
    def __init__(self, cls_num=10):
        super(ResNetClassifier, self).__init__()
        self.cls_num = cls_num
        self.classifier = nn.Linear(512, cls_num)
        
    def forward(self, x):
        output = self.classifier(x)
        return output 


class Net(nn.Module):
    def __init__(self, cls_num=5):
        super(Net, self).__init__()
        self.cls_num = cls_num
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, self.cls_num)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(F.relu(x))
        x = self.dropout(x)
        x = x.view(-1, 14 * 14 * 64)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
#         output = F.softmax(x, dim=1)
#         print(output)
        return output

    def get_mid_feat(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(F.relu(x))
        x = self.dropout(x)
        x = x.view(-1, 14 * 14 * 64)
        x = F.relu(self.fc1(x))
        
        return x
    

def build_CNN(basemodel_name, pretrained=False):
    if basemodel_name=='ResNet':
        base_model = models.resnet18(pretrained=pretrained)
        return ResNetCNN(base_model)
    elif basemodel_name=='AlexNet':
        base_model = models.alexnet(pretrained=pretrained)
        return AlexNetCNN(base_model)
    
def build_Classifier(basemodel_name, cls_num=10, pretrained=False):
    if basemodel_name=='ResNet':
        return ResNetClassifier(cls_num)
    elif basemodel_name=='AlexNet':
        base_model = models.alexnet(pretrained=pretrained)
        return AlexNetClassifier(base_model, cls_num)