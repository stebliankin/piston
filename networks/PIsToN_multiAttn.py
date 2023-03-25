"""
Objective:
    The main PIsToN model.
    The network efficiently combines the ViT and Hybrid components to include spatial and attention features.
    Each branch contains an independent transformer network that learns separate latent
     vectors with spatial attention maps. The latent vectors of each branch are then aggregated with a class token into
     the transformer encoder for a final prediction. Since the interatomic distances affect the energies from each group,
     the patch distance feature was appended as an extra channel to each branch.
     The class probability was computed as the Euclidean distance to the class centroid.

Author:
    Vitalii Stebliankin (vsteb002@fiu.edu)
    Bioinformatics Research Group (BioRG)
    Florida International University

"""

import torch
from torch import nn
from networks.ViT_pytorch import Encoder
from networks.ViT_hybrid import ViT_Hybrid_encoder
from losses.proto_loss import ProtoLoss
from losses.supCon_loss import SupConLoss

class PIsToN_multiAttn(nn.Module):

    def __init__(self, config, img_size=24, num_classes=2, zero_head=False, margin=0, temperature=0.5):
        super(PIsToN_multiAttn, self).__init__()

        """
        Input: Image with the following features:

        0 - Shape index | p1
        1 - Distance depended curvature | p1
        2 - Hydrogen bond potential | p1
        3 - Charge | p1
        4 - Hydrophobicity | p1
        5 - Shape index | p1
        6 - Distance depended curvature | p1
        7 - Hydrogen bond potential | p1
        8 - Charge | p1
        9 - Hydrophobicity | p1
        10 - Distance between atoms | p1 and p2
        11 - Relative ASA | p1
        12 - Relative ASA | p1

        Energies with the following features:
        0  - glob      - global score of the candidate, which is linear combination of the terms described bellow. To rank the candidates, you should sort the rows by this column in ascending order.
        1  - aVdW      - attractive van der Waals
        2  - rVdW      - repulsive van der Waals
        3  - ACE       - Atomic Contact Energy | desolvation (10.1006/jmbi.1996.0859)
        4  - inside    - "Insideness" measure, which reflects the concavity of the interface.
        5  - aElec     - short-range attractive electrostatic term
        6  - rElec     - short-range repulsive electrostatic term
        7  - laElec    - long-range attractive electrostatic term
        8  - lrElec    - long-range repulsive electrostatic term
        9  - hb        - hydrogen and disulfide bonding
        10 - piS	     - pi-stacking interactions
        11 - catpiS	  - cation-pi interactions
        12 - aliph	  - aliphatic interactions

        Loss = alpha * (BCE_i/5) + Global_BCE, where alpha is a hyperparameter from 0 to 1

        """

        # First tuple is the feature index in the image, and the second tuple is the feature index of energy terms
        self.index_dict = {
            'shape_complementarity': ( (0, 5, 1, 6, 10), (1,2,4) ),
            'RASA': ( (11, 12, 10), (3,) ),
            'hydrogen_bonds': ( (2, 7, 10), (9,) ),
            'charge': ( (3, 8, 10), (5, 6, 7, 8, 10, 11, 12) ),
            'hydrophobicity': ((4, 9, 10), ()),
        }

        self.img_size = img_size
        self.num_classes = num_classes
        self.zero_head = zero_head


        self.spatial_transformers_list = nn.ModuleList()

        for feature in self.index_dict.keys():
            self.spatial_transformers_list.append(self.init_transformer(config, channels=len(self.index_dict[feature][0]),
                                                                n_individual=len(self.index_dict[feature][1])))


        self.classifier = config.classifier
        self.feature_transformer = Encoder(config, vis=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size), requires_grad=True)
        self.proto_vector_pos = nn.Parameter(torch.rand(1,config.hidden_size), requires_grad=True)
        self.proto_vector_neg = nn.Parameter(torch.rand(1,config.hidden_size), requires_grad=True)
        self.margin = margin
        self.temperature = temperature
        #self.cosine_loss = CosineEmbeddingLoss(margin=margin)

        # self.proto_loss_fn = ProtoLoss(margin=margin, prototype_activation_function=prototype_activation_function)

        #self.final_head = nn.Linear(config.hidden_size, num_classes)

    def init_transformer(self, config, channels, n_individual):
        """
        Initialize Transformer Network for a given tupe of features
        :param model_config:
        :param channels:
        :param n_individual:
        :return:
        """
        return ViT_Hybrid_encoder(config, n_individual, img_size=self.img_size,
                        num_classes=self.num_classes, channels=channels, vis=True)


    def forward(self, img, energies, labels=None):

        all_x = []
        all_spatial_attn = []
        for i, feature in enumerate(self.index_dict.keys()):
            img_tmp = img[:,self.index_dict[feature][0],:,:]
            energy_tmp = energies[:, self.index_dict[feature][1]]

            x, attn = self.spatial_transformers_list[i](img_tmp, energy_tmp)

            all_x.append(x)
            all_spatial_attn.append(attn)

        x = torch.stack(all_x, dim=1) # (36, 5, 16)

        # Create classification token
        B = x.shape[0] #batch
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        x, feature_attn = self.feature_transformer(x)

        x = x[:, 0]
        x = nn.functional.normalize(x)  # L2 normalization

        proto_vector_pos = nn.functional.normalize(self.proto_vector_pos)
        proto_vector_neg = nn.functional.normalize(self.proto_vector_neg)

        dist = nn.PairwiseDistance()
        dist_to_pos_prototype = dist(x, proto_vector_pos.repeat(x.shape[0], 1))
        dist_to_neg_prototype = dist(x, proto_vector_neg.repeat(x.shape[0], 1))

        # 0 - minimize distance to positives
        logits = torch.stack([dist_to_pos_prototype-dist_to_neg_prototype, dist_to_neg_prototype-dist_to_pos_prototype], axis=1)

        scores = dist_to_pos_prototype-dist_to_neg_prototype
        #scores = dist_to_pos_prototype - 1
        #scores =
        #print(labels)
        #print(scores)
        # print(dist_to_pos_prototype)
        # print(logits)
        # print(dist_to_pos_prototype)
        # print()
        # print(dist_to_neg_prototype)
        # print()
        if labels is not None:
            proto_loss_fn = ProtoLoss(margin=self.margin)
            bce_loss_fn = nn.CrossEntropyLoss()
            SupConLoss_fn = SupConLoss(temperature=self.temperature, base_temperature=self.temperature*10)

            proto_loss = proto_loss_fn(x, proto_vector_pos, proto_vector_neg, labels)
            BCE_loss = bce_loss_fn(logits, labels) # probability of the positive class
            supCon_loss = SupConLoss_fn(x, labels)
            loss = proto_loss + BCE_loss + supCon_loss

        # Do the Supervised Contrastive Loss jointly with Cosine Loss
        #x = self.final_head(x)

            return scores, (all_spatial_attn, feature_attn), loss
        else:
            return scores, (all_spatial_attn, feature_attn)

