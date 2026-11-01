import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class LatentHelper(torch.nn.Module):
    def __init__(self, layers, latent_index):
        super().__init__()
        self.layers = layers
        self.model_before = torch.nn.Sequential(*(layers[:latent_index]))
        self.model_after = torch.nn.Sequential(*(layers[latent_index:]))
        self.latent = None

    def forward(self, x):
        x = self.model_before(x)
        self.latent = x
        return self.model_after(x)