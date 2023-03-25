"""
Objective:
   Dataset object to read interface maps and energy terms

Author:
    Vitalii Stebliankin (vsteb002@fiu.edu)
    Bioinformatics Research Group (BioRG)
    Florida International University

"""
from torch.utils.data import Dataset
import numpy as np
import random
import os

import torch

from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly
#
# from collections import defaultdict

# import torchvision.transforms as transforms
from scipy import ndimage


def read_energies(energies_dir, ppi):
    """
        (0) - indx
        (1) - Lrmsd     - ligand rmsd of the final position, after the rigid-body optimization.
        (2) -Irmsd     - interface rmsd of the final position, after the rigid-body optimization.
        (3) - st_Lrmsd  - initial ligand rmsd.
        (4) - st_Irmsd  - initial ligand rmsd.
    0 - (5) - glob      - global score of the candidate, which is linear combination of the terms described bellow. To rank the candidates, you should sort the rows by this column in ascending order.
    1 - (6) - aVdW      - attractive van der Waals
    2 - (7) - rVdW      - repulsive van der Waals
    3 - (8) - ACE       - Atomic Contact Energy | desolvation (10.1006/jmbi.1996.0859)
    4 - (9) - inside    - "Insideness" measure, which reflects the concavity of the interface.
    5 - (10) - aElec     - short-range attractive electrostatic term
    6 - (11) - rElec     - short-range repulsive electrostatic term
    7 - (12) - laElec    - long-range attractive electrostatic term
    8 - (13) - lrElec    - long-range repulsive electrostatic term
    9 - (14) - hb        - hydrogen and disulfide bonding
    10 - (15) - piS	     - pi-stacking interactions
    11 - (16) - catpiS	  - cation-pi interactions
    12 - (17) - aliph	  - aliphatic interactions
         (18) - prob      - rotamer probability
    :param ppi:
    :return:
    """
    energies_path = f"{energies_dir}/refined-out-{ppi}.ref"

    to_read = False
    all_energies = None

    with open(energies_path, 'r') as f:
        for line in f.readlines():
            if to_read:
                all_energies = line.split('|')
                all_energies = [x.strip(' ') for x in all_energies]
                all_energies = all_energies[5:18]
                all_energies = [float(x) for x in all_energies]
                all_energies = np.array(all_energies)
                break
            if 'Sol # |' in line:
                to_read = True
    if all_energies is None:
        # energies couldn't be computed. Assign them to zero
        all_energies = np.zeros(13)

    all_energies = np.nan_to_num(all_energies)
    return all_energies

def learn_background_mask(grid):
    """
    Returns the mask with zero elements outside the patch
    :param grid: example of a grid image
    :return: mask
    """
    mask = np.zeros((grid.shape[0], grid.shape[1]))
    radius = grid.shape[0] / 2
    for row_i in range(grid.shape[0]):
        for column_i in range(grid.shape[1]):
            # Check if coordinates are within the radius
            x = column_i - radius
            y = radius - row_i
            if x ** 2 + y ** 2 <= radius ** 2:
                mask[row_i][column_i] = 1
    return mask

class PDB_complex_training(Dataset):
    # Including FireDock energies

    def __init__(self, ppi_list, training_mode, data_prepare_dir,
                 energies_mean, energies_std, neg_pos_ratio=1,
                 std=None, mean=None, feature_subset=None,):
        """
        Dataset instance
        :param ppi_list:
        :param training_mode (bool):
        :param docked_dir:
        :param pos_grid_dir:
        :param energies_dir: directory with pre-computed FireDock energies
        :param original_feats_only:
        :param neg_pos_ratio:
        :param std: array with of size equals to number of features. Each value is the distribution of STD for each feature;
                    If None, don't perform standard scaling.
        :param mean: array with of size equals to number of features. Each value is the distribution of mean for each feature;
                    If None, don't perform standard scaling
        """
        self.ppi_list = ppi_list
        self.training_mode = training_mode
        self.docked_dir = data_prepare_dir + '/docked/'
        self.data_prepare_dir = data_prepare_dir
        self.pos_grid_dir = data_prepare_dir + '/07-grid/'
        self.neg_pos_ratio = neg_pos_ratio
        self.feature_subset = feature_subset
        #energy_tmp = self.read_energies(self.ppi_list[0], docked_flag=False)

        self.pos_dict, self.neg_dict = self.select_pos_neg(ppi_list)
        self.std = std
        self.mean = mean
        self.energies_mean = energies_mean
        self.energies_std = energies_std
        count_pos = sum([len(self.pos_dict[x]) for x in self.pos_dict.keys()]) + len(
            self.pos_dict.keys())  # + each true complex will be used as positive
        count_neg = sum([len(self.neg_dict[x]) for x in self.neg_dict.keys()])

        print(
            f"Dataset is constructed from {self.data_prepare_dir} directory. Extracted {count_pos} positive and {count_neg} negative images from {len(self.ppi_list)} complexes.")

        if self.feature_subset is not None:
            print(f"Using the following subset of features: {self.feature_subset}")
        # load random image

        random_image = np.load(self.pos_grid_dir + random.choice(self.ppi_list) + '.npy', allow_pickle=True)
        self.background_mask = learn_background_mask(random_image)
        self.n_features = random_image.shape[-1]

    def __len__(self):
        return len(self.ppi_list)

    def select_pos_neg(self, ppi_list):
        pos_dict = {}
        neg_dict = {}
        ppi_list_updated = []
        for ppi in ppi_list:
            iRMSD_file = self.docked_dir + ppi + '/irmsd.csv'
            pos_docked_ppis = []
            neg_docked_ppis = []
            if os.path.exists(iRMSD_file):
                with open(iRMSD_file, 'r') as f:
                    f.readline()
                    for line in f.readlines():
                        model_i, model_ppi_i, irmsd, lrmsd, fnat = line.strip('\n').split(',')
                        if not os.path.exists(
                                self.docked_dir + ppi + '/07-grid/' + model_ppi_i + '.npy'):  # skip if grid doesn't exist
                            continue
                        elif float(fnat) >= 0.1 and (float(lrmsd) <= 10 or float(irmsd) <= 4):
                            pos_docked_ppis.append(model_ppi_i)
                        else:
                            neg_docked_ppis.append(model_ppi_i)

            if len(neg_docked_ppis) > 0: #and len(pos_docked_ppis)>0:
                pos_dict[ppi] = pos_docked_ppis
                neg_dict[ppi] = neg_docked_ppis
                ppi_list_updated.append(ppi)

        self.ppi_list = ppi_list_updated
        return pos_dict, neg_dict

    def rotate(self, grid):
        # randomly rotate the patch
        angle = np.random.randint(low=1, high=360)

        for feature_i in range(0, grid.shape[-1]):
            grid[:, :, feature_i] = ndimage.rotate(grid[:, :, feature_i], angle, reshape=False)

        # if self.vis_patch:
        #     print("Complex: {}; Label: {}".format(ppi, self.labels[i]))
        #     print('Rotated on angle {}'.format(angle))
        #     self.vis_patch_f(grid, self.labels[i], ppi)

        return grid

    def __getitem__(self, i):
        ppi = self.ppi_list[i]

        true_grid1 = self.rotate(np.load(self.pos_grid_dir + ppi + '.npy', allow_pickle=True))
        true_grid2 = self.rotate(np.load(self.pos_grid_dir + ppi + '.npy', allow_pickle=True))


        total_n_negatives = min((len(self.pos_dict[ppi]) + 2) * self.neg_pos_ratio, len(
            self.neg_dict[ppi]))  # +2 because we will add native complex as the positive example
        total_n_positives = int(total_n_negatives / self.neg_pos_ratio)

        # Shuffle both dictionaries
        pos_complexes, neg_complexes = self.pos_dict[ppi], self.neg_dict[ppi]

        if self.training_mode:
            random.shuffle(pos_complexes)
            random.shuffle(neg_complexes)

        pos_complexes, neg_complexes = pos_complexes[:total_n_positives - 1], neg_complexes[:total_n_negatives]
        pos_paths = [f"{self.docked_dir}/{ppi}/07-grid/{x}.npy" for x in pos_complexes]
        neg_paths = [f"{self.docked_dir}/{ppi}/07-grid/{x}.npy" for x in neg_complexes]

        pos_grids = [self.rotate(np.load(pos_path, allow_pickle=True)) for pos_path in pos_paths] + [true_grid1, true_grid2]
        neg_grids = [self.rotate(np.load(neg_path, allow_pickle=True)) for neg_path in neg_paths]

        energy_dir_pos = f"{self.data_prepare_dir}/07-grid/"
        energy_dir_docked = f"{self.docked_dir}/{ppi}/07-grid/"
        pos_energies = [read_energies(energy_dir_docked, x) for x in pos_complexes] + \
                       [read_energies(energy_dir_pos, ppi), read_energies(energy_dir_pos, ppi)]

        neg_energies = [read_energies(energy_dir_docked, x) for x in neg_complexes]

        # if not docked_flag:
        #     energies_path = f"{self.data_prepare_dir}/07-grid/refined-out-{model_ppi}.ref"
        # else:
        #     energies_path = f"{self.docked_dir}/{true_ppi}/07-grid/refined-out-{model_ppi}.ref"

        labels = np.array([1] * len(pos_grids) + [0] * len(neg_grids))
        grid = pos_grids + neg_grids
        all_energies = np.stack(pos_energies+neg_energies)
        grid = np.swapaxes(grid, -1, 1).astype(np.float32)

        if self.feature_subset:
            grid = grid[:, self.feature_subset, :, :]  # subset of features
        if self.mean is not None and self.std is not None:
            # Perform standard scaling
            for feature_i in range(grid.shape[1]):
                grid[:, feature_i, :, :] = (grid[:, feature_i, :, :] - self.mean[feature_i]) / self.std[feature_i]
            # Mask out values that are out of the radius:
            grid = np.logical_and(grid, self.background_mask) * grid
        for energy_i in range(all_energies.shape[1]):
            all_energies[:,energy_i] = (all_energies[:,energy_i] - self.energies_mean[energy_i])/self.energies_std[energy_i]

        return grid, all_energies, labels, ppi



class PISToN_dataset(Dataset):
    def __init__(self, grid_dir, ppi_list, attn=None):

        ### Empirically learned mean and standard deviations:
        mean_array = [0.06383528408485302, 0.043833505848899605, -0.08456032982438057, 0.007828608135306595,
                      -0.06060602411612203, 0.06383528408485302, 0.043833505848899605, -0.08456032982438057,
                      0.007828608135306595, -0.06060602411612203, 11.390402735801011, 0.1496338245579665,
                      0.1496338245579665]
        std_array = [0.4507792893174703, 0.14148081793902434, 0.16581325050002976, 0.28599861830017204,
                     0.6102229371168204, 0.4507792893174703, 0.14148081793902434, 0.16581325050002976,
                     0.28599861830017204, 0.6102229371168204, 7.265311558033949, 0.18003612950610695,
                     0.18003612950610695]

        all_energies_mean = [-193.1392953586498, -101.97838818565408, 264.2099535864983, -17.27086075949363,
                             16.329959915611877, -102.78101054852341, 36.531006329113836, -27.1124789029536,
                             16.632626582278455, -8.784924050632918, -6.206329113924051, -1.8290084388185655,
                             -11.827215189873417]
        all_energies_std = [309.23521244706757, 66.75799437657368, 9792.783784373369, 25.384427268309658,
                            7.929941961525389, 94.05055841984323, 47.22518557457095, 24.392679889433445,
                            17.57399925906454, 7.041949880295568, 6.99554122803362, 2.557571754303165,
                            13.666329541281653]

        all_grids = []
        all_energies = []

        ppi_to_idx = {} # map ppi id to idx

        i=0
        for ppi in ppi_list:
            if os.path.exists(f"{grid_dir}/{ppi}.npy"):
                ppi_to_idx[ppi] = i
                grid = np.load(f"{grid_dir}/{ppi}.npy", allow_pickle=True)
                all_grids.append(grid)
                #energies_path = f"{grid_dir}/refined-out-{ppi}.ref"
                energies = read_energies(grid_dir, ppi)
                all_energies.append(energies)
                i+=1
        self.ppi_to_idx = ppi_to_idx


        background_mask = learn_background_mask(grid)

        grid = np.stack(all_grids, axis=0)
        grid = np.swapaxes(grid, -1, 1).astype(np.float32)
        all_energies = np.stack(all_energies, axis=0)

        print(f"Interaction maps shape: {grid.shape}")
        print(f"All energies shape: {all_energies.shape}")

        ### Standard scaling

        # Interactino maps:
        for feature_i in range(grid.shape[1]):
            grid[:, feature_i, :, :] = (grid[:, feature_i, :, :] - mean_array[feature_i]) / std_array[feature_i]
            # Mask out values that are out of the radius:
            grid = np.logical_and(grid, background_mask) * grid

        ## ENERGIES:
        for energy_i in range(all_energies.shape[1]):
            all_energies[:, energy_i] = (all_energies[:, energy_i] - all_energies_mean[energy_i]) / all_energies_std[
                energy_i]

        self.grid = grid
        self.all_energies = all_energies
        self.grid_dir = grid_dir
        self.ppi_list = ppi_list

    def vis_patch(self, ppi, html_path=None, attn=None):
        feature_pairs = {
            'shape_index': (0, 5),
            'ddc': (1, 6),
            'electrostatics': (2, 7),
            'charge': (3, 8),
            'hydrophobicity': (4, 9),
            'RASA': (11, 12),
            'patch_dist': (10,),
        }
        grid_dir = self.grid_dir
        resnames_path = grid_dir + ppi + '_resnames.npy'
        patch_path = grid_dir + ppi + '.npy'
        patch_np = np.load(patch_path, allow_pickle=True)

        patch_resnames = np.load(resnames_path, allow_pickle=True)
        # patch_resnames = patch_resnames[:,:,0]
        n_feat = int(patch_np.shape[-1] / 2)
        key_names = list(feature_pairs.keys())
        fig = make_subplots(2, n_feat,
                            subplot_titles=key_names[:n_feat])

        patch_dist = patch_np[:, :, feature_pairs['patch_dist']].reshape((patch_np.shape[0], patch_np.shape[1]))
        patch_dist = np.round(patch_dist, 2)
        for col_i in range(n_feat):
            for row_i, pair_i in enumerate(feature_pairs[key_names[col_i]]):
                patch_i = patch_np[:, :, pair_i]
                if attn is not None:
                    #mask = (attn+0.01)*(attn==0) + attn
                    mask = (attn>0) * attn
                    patch_i = patch_i * mask

                customdata = np.stack([patch_resnames[:, :, row_i], patch_dist], axis=-1)

                fig.add_trace(go.Heatmap(
                    z=patch_i,
                    customdata=customdata,
                    hovertemplate='<b>Value:%{z:.3f}</b><br>Amino Acid:%{customdata[0]}; dist:%{customdata[1]}',
                    name='',
                    colorscale='RdBu',
                    zmid=0,
                    showscale=False,
                    showlegend=False
                )
                    ,
                    row_i + 1, col_i + 1)
        fig.update_layout(
            title_text='The interactive patch pair for {}. Hover to see the value and corresponding amino acid name.'.format(
                ppi))
        fig.show()
        if html_path is not None:
            plotly.offline.plot(fig, filename=html_path)

    def __len__(self):
        return self.grid.shape[0]

    def read_scaled(self, ppi, device):
        idx = self.ppi_to_idx[ppi]
        grid = torch.from_numpy(np.expand_dims(self.grid[idx], 0))
        energies = torch.from_numpy(np.expand_dims(self.all_energies[idx], 0))
        return grid.to(device), energies.float().to(device)


    def __getitem__(self, idx):
        return self.grid[idx], self.all_energies[idx]