import torch
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision
from torch import nn, Tensor
import dataset
import pdb


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = {}
        self.names = {}
        self.layer_numbers = 0
        self.register_hook(model)

    def register_hook(self,model):
        idx = 0
        for name,layer in model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer,BatchNorm2d) or isinstance(layer,nn.ReLU):
                layer.register_forward_hook(self.get_activation(idx))
                self.names[idx] = name
                idx += 1
        self.layer_numbers = idx

    def get_activation(self,name):
        def hook_fn(module,input,output):
            self.activations[name] = output
        return hook_fn

    def forward(self, x):
        return self.model(x)
        
