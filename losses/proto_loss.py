"""
Objective:
    Margin Ranking Loss that guids the embeddings clustering.

Author:
    Vitalii Stebliankin (vsteb002@fiu.edu)
    Bioinformatics Research Group (BioRG)
    Florida International University
"""

import torch
import torch.nn as nn


class ProtoLoss(nn.Module):
    """
    The contrastive loss function is aiming to minimize the distance between embeddings and prototypes of its own class,
    while maximizing it for instances of different class.
    The prototype_vectors_pos and prototype_vectors_neg should be part of the architecture.
    """

    def __init__(self, margin=0.3, prototype_activation_function='linear', reduction='mean'):
        super(ProtoLoss, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)
        self.prototype_activation_function = prototype_activation_function
        self.epsilon = 1e-5

    def forward(self, features, prototype_vectors_pos, prototype_vectors_neg, labels):
        """
        :param features: latent vector from the mini-batch
        :param prototype_vectors_pos: prototype vector of the positive class
        :param prototype_vectors_neg: prototype vector of the negative class
        :param labels: ground truth vector from the
        :return:
        """

        dist = nn.PairwiseDistance()
        dist_to_pos_prototype = dist(features, prototype_vectors_pos.repeat(features.shape[0], 1))
        dist_to_neg_prototype = dist(features, prototype_vectors_neg.repeat(features.shape[0], 1))

        pos_mask = torch.eq(labels, 1)
        neg_mask = ~pos_mask

        dist_same_class = self.distance_2_similarity(dist_to_pos_prototype * pos_mask + dist_to_neg_prototype * neg_mask)
        dist_opposite_class = self.distance_2_similarity(dist_to_pos_prototype*neg_mask + dist_to_neg_prototype*pos_mask)

        y = torch.ones_like(dist_opposite_class) # y=1 means maximize the first input and minimize the second one
        return self.ranking_loss(dist_same_class, dist_opposite_class, -y)

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return -torch.log( (distances + 1) / (distances + self.epsilon) )
        elif self.prototype_activation_function == 'linear':
            return distances