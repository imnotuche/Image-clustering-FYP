import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet-50 
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # We remove the last layer (the classifier) 
        # because we want the "features," not a 1000-class prediction 
        self.feature_layer = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # Extract features and flatten them into a vector 
        with torch.no_grad(): # We don't need to train ResNet, just use it
            features = self.feature_layer(x)
        return features.view(features.size(0), -1) # Returns a vector of size 2048