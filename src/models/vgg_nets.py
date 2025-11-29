import torch
import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self):
        '''
            VGG16 Architecture (Features)
            
            (0): Conv1_1
            (1): ReLU1_1
            (2): Conv1_2
            (3): ReLU1_2
            (4): MaxPool1

            (5): Conv2_1
            (6): ReLU2_1
            (7): Conv2_2
            (8): ReLU2_2
            (9): MaxPool2

            (10): Conv3_1
            (11): ReLU3_1
            (12): Conv3_2
            (13): ReLU3_2
            (14): Conv3_3
            (15): ReLU3_3
            (16): MaxPool3

            (17): Conv4_1
            (18): ReLU4_1
            (19): Conv4_2
            (20): ReLU4_2
            (21): Conv4_3
            (22): ReLU4_3
            (23): MaxPool4

            (24): Conv5_1
            (25): ReLU5_1
            (26): Conv5_2
            (27): ReLU5_2
            (28): Conv5_3
            (29): ReLU5_3
            (30): MaxPool5
        
        '''
        super().__init()
        self.vgg = models.vgg16(pretrained=True).features
        
        self.style_layers = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        
        self.content_layer = {
            'relu4_2': 20
        }
        
        self.selected_indices = [3, 8, 15, 20, 22]
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        feature_maps = {}
        
        for index, layer in enumerate(self.vgg):
            x = layer(x)
            
            if index in self.selected_indices:
                for name, i in {**self.style_layers, **self.content_layer}.items():
                    if index == i:
                        feature_maps[name] = x

        return feature_maps                        

class VGG19(nn.Module):
    '''
        VGG19 Architecture (Features)
    
        (0): Conv1_1
        (1): ReLU1_1
        (2): Conv1_2
        (3): ReLU1_2
        (4): MaxPool1
        
        (5): Conv2_1
        (6): ReLU2_1
        (7): Conv2_2
        (8): ReLU2_2
        (9): MaxPool2
        
        (10): Conv3_1
        (11): ReLU3_1
        (12): Conv3_2
        (13): ReLU3_2
        (14): Conv3_3
        (15): ReLU3_3
        (16): Conv3_4
        (17): ReLU3_4
        (18): MaxPool3

        (19): Conv4_1
        (20): ReLU4_1
        (21): Conv4_2
        (22): ReLU4_2
        (23): Conv4_3
        (24): ReLU4_3
        (25): Conv4_4
        (26): ReLU4_4
        (27): MaxPool4

        (28): Conv5_1
        (29): ReLU5_1
        (30): Conv5_2
        (31): ReLU5_2
        (32): Conv5_3
        (33): ReLU5_3
        (34): Conv5_4
        (35): ReLU5_4
        (36): MaxPool5
    '''
    def __init__(self, use_relu=False):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features
        # print(features)
        
        if use_relu:
            self.style_layers = {
                'relu1_1': 1,
                'relu2_1': 6, 
                'relu3_1': 11, 
                'relu4_1': 20, 
                'relu5_1': 29
            }
            
            self.content_layer = {
                'relu4_2': 22
            }
            
            self.selected_indices = [1, 6, 11, 20, 22, 29]
            
        else:
            self.style_layers = {
                'conv1_1': 0,
                'conv2_1': 5,
                'conv3_1': 10,
                'conv4_1': 19,
                'conv5_1': 28
            }
            
            self.content_layer = {
                'relu4_2': 22
            }
            
            self.selected_indices = [0, 5, 10, 19, 22, 28]
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        
        feature_maps = {}
        
        for index, layer in enumerate(self.vgg):
            x = layer(x)
            
            if index in self.selected_indices:
                for name, i in {**self.style_layers, **self.content_layer}.items():
                    if i == index:
                        feature_maps[name] = x        

        return feature_maps                