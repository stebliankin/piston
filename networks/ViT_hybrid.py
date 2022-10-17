"""
Objective:
   The PIsToN-Hybrid component.
   The method combines empirically computed energies with the interface maps.
    The $Q$ energy terms are projected on to a latent vector using a fully connected network (FC),
                                        which is then concatenated to the vector obtained from ViT

Author:
    Vitalii Stebliankin (vsteb002@fiu.edu)
    Bioinformatics Research Group (BioRG)
    Florida International University

"""

import torch
from torch import nn
from .ViT_pytorch import Transformer

class ViT_Hybrid(nn.Module):
    def __init__(self, config, n_individual, img_size=24, num_classes=2, zero_head=False, vis=False, channels=13):
        super(ViT_Hybrid, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, channels, vis)
        self.individual_nn = nn.Linear(n_individual, n_individual)

        self.combine_nn = nn.Linear(config.hidden_size + n_individual, config.hidden_size)

        self.classifier_nn = nn.Linear(config.hidden_size, num_classes)

        self.af_ind = nn.GELU()
        self.af_combine = nn.GELU()


    def forward(self, x, individual_feat):
        x, attn_weights = self.transformer(x)
        x = x[:, 0] # classification token

        individual_x = self.individual_nn(individual_feat)
        individual_x = self.af_ind(individual_x)

        x = torch.cat([x, individual_x], dim=1)

        x = self.combine_nn(x)
        x = self.af_combine(x)

        logits = self.classifier_nn(x)

        return logits, attn_weights

class ViT_Hybrid_encoder(nn.Module):
    def __init__(self, config, n_individual, img_size=24, num_classes=2, zero_head=False, vis=False, channels=13):
        super(ViT_Hybrid_encoder, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, channels, vis)
        self.individual_nn = nn.Linear(n_individual, n_individual)

        self.combine_nn = nn.Linear(config.hidden_size + n_individual, config.hidden_size)

        self.af_ind = nn.GELU()
        self.af_combine = nn.GELU()


    def forward(self, x, individual_feat):
        x, attn_weights = self.transformer(x)
        x = x[:, 0] # classification token

        individual_x = self.individual_nn(individual_feat)
        individual_x = self.af_ind(individual_x)

        x = torch.cat([x, individual_x], dim=1)

        x = self.combine_nn(x)
        x = self.af_combine(x)

        #logits = self.head(x[:, 0])

        return x, attn_weights

