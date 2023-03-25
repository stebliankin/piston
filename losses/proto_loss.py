"""
Objective:
    Margin Ranking Loss guids the embeddings clustering.
    The code was adapted from ProtoPNet:
    https://github.com/cfchen-duke/ProtoPNet/blob/81bf2b70cb60e4f36e25e8be386eb616b7459321/model.py
Author:
    Vitalii Stebliankin (vsteb002@fiu.edu)
    Bioinformatics Research Group (BioRG)
    Florida International University
"""

import torch
from torch import ones_like

class ProtoLoss(torch.nn.Module):
    """
    The centroid loss introduces the cluster centroid
    The centroid_pos and centroid_neg should be part of the architecture.
    """

    def __init__(self, margin=0, centroid_active_fun='linear', reduction='mean'):
        super(ProtoLoss, self).__init__()
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin, reduction=reduction)
        self.centroid_active_fun = centroid_active_fun
        self.eps = 0.00001

    def forward(self, emb, centroid_pos, centroid_neg, labels):
        """
        :param emb: embedding vector from the model
        :param centroid_pos: cluster centroid of the positive class
        :param centroid_neg: cluster centroid of the negative class
        :param labels: ground truth vector from the
        :return: the Margin Ranking loss
        """

        dist = torch.nn.PairwiseDistance()
        dist_to_pos_centroid = dist(emb, centroid_pos.repeat(emb.shape[0], 1))
        dist_to_neg_centroid = dist(emb, centroid_neg.repeat(emb.shape[0], 1))
        mask_pos = torch.eq(labels, 1)
        mask_neg = ~mask_pos
        proto_active_same = self.dist2sim(mask_pos*dist_to_pos_centroid + mask_neg*dist_to_neg_centroid)
        proto_active_opposite = self.dist2sim(mask_neg*dist_to_pos_centroid + mask_pos*dist_to_neg_centroid)
        mode = ones_like(proto_active_opposite) # mode=1 will maximize the first input while minimize the second
        return self.ranking_loss(proto_active_opposite, proto_active_same, mode)

    def dist2sim(self, dist):
        if self.centroid_active_fun == 'linear':
            return dist
        elif self.centroid_active_fun == 'log':
            return -torch.log( (dist + 1) / (dist + self.eps) )
