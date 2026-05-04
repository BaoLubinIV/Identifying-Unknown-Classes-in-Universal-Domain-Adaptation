import lightning as L
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchmetrics import Accuracy, StatScores
from scipy.stats import entropy
from torchvision.models import resnet50
import clip
import math
import os
from utils import HScore, CrossEntropyLabelSmooth, CustomLRScheduler

from augmentation import get_tta_transforms


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class CLIPBlock(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32"):
        super(CLIPBlock, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name)
        self.output_dim = 512
        
        self.clip_model = self.clip_model.to(self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(self, x, tta_transforms = False):
        if isinstance(x, list):  
            x = torch.stack([self.clip_preprocess(img) for img in x]).to(self.device)  
        else:
            x = self.clip_preprocess(x).to(self.device)
        if tta_transforms:
            aug_tta_transform = get_tta_transforms()
            x = aug_tta_transform(x)
            
        with torch.no_grad(): 
            x = self.clip_model.encode_image(x)
            x = x.float()
        return x
        
    def encode_text(self, texts):
        with torch.no_grad():
            return self.clip_model.encode_text(texts).float()

    def tokenize(self, texts):
        return clip.tokenize(texts, truncate=True)
        
class Resnet(nn.Module):
    def __init__(self, resize_size=256, crop_size=224):
        super(Resnet, self).__init__()
        model_resnet = resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.output_dim = model_resnet.fc.in_features
        
        # Define preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x, tta_transforms = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.stack([self.preprocess(img) for img in x]).to(device)
        if tta_transforms:
            aug_tta_transform = get_tta_transforms()
            x = aug_tta_transform(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x



class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim=256, type='ori'):
        super(FeatureExtractor, self).__init__()
        self.feature_dim = feature_dim
        self.dense_layer = nn.Linear(input_dim, feature_dim)
        self.relu = nn.ReLU()
        self.dense_layer.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = x.float()
        x = self.dense_layer(x)
        x = self.relu(x)
        return x
        
class Classifier(nn.Module):
    def __init__(self, class_num, feature_dim=256, type='linear'):
        super(Classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(feature_dim, class_num), name='weight')
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(feature_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class MLPMapping(nn.Module):
    def __init__(self, input_dim=512, output_dim=768, prefix_length=10, hidden_dim=1024):
        super(MLPMapping, self).__init__()
        self.prefix_length = prefix_length
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * prefix_length)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out.view(-1, self.prefix_length, self.output_dim)