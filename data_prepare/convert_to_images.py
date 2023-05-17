import pymesh #Importing pymesh here avoids library conflict (CXXABI_1.3.11)
from tqdm import tqdm
import numpy as np
import pdb
from sklearn.neighbors import KDTree
import os
from scipy import ndimage

from Bio.PDB import PDBParser, DSSP

from data_prepare.map_patch_atom import map_patch_indices

def polar_to_cartesian(rho, theta, rotate_theta=0):
    # Interpolate the polar coordinates into a d x d square, where d is the diameter of the patch
    # rotate_theta - rotate all coordinates on a constant angle (used to search for matching patches).
    #cart_grid = np.zeros((rho.shape[0],radius*2,radius*2))
    cart_coord_x = np.zeros(rho.shape)
    cart_coord_y = np.zeros(rho.shape)

    for coord_i in range(0, rho.shape[0]):
        rho_coord = rho[coord_i]
        theta_coord = theta[coord_i]
        cart_coord_x[coord_i] = rho_coord*np.cos(theta_coord+rotate_theta)
        cart_coord_y[coord_i] = rho_coord*np.sin(theta_coord+rotate_theta)

    return cart_coord_x, cart_coord_y

def get_new_coord_patch(radius):
    new_patch_coord = []
    for i in range(0, radius*2):
        for j in range(0, radius*2):
            new_patch_coord.append((i-radius,j-radius))
    return np.array(new_patch_coord)


def compute_patch_grid(x, y, input_feat, radius, interpolate=True, stringarray=False):
    old_coord = np.stack((x,y), axis=-1)
    if not stringarray:
        patch_grid = np.zeros((radius*2, radius*2, input_feat.shape[1])) # shape = (24 x 24 x 5)
    else:
        patch_grid = np.array(radius*2*[np.array(['x' for x in range(radius*2)], dtype=object)])
        patch_grid = np.expand_dims(patch_grid, axis=-1)

    for feature_i in range(0, patch_grid.shape[-1]):
        #print("[{}] Computing grid for feature {}".format(datetime.now(), feature_i))
        old_coord_patch = old_coord
        new_coord_patch = get_new_coord_patch(radius) # grid coordinates [-r, r]
        # map old coordinates to the new grid:
        kdt = KDTree(old_coord_patch)
        if interpolate:
            dist, indx_old = kdt.query(new_coord_patch, k=4) #interpolate across 4 nearest neighbors
        else:
            dist, indx_old = kdt.query(new_coord_patch, k=1)
        # Square the distances (as in the original pyflann)
        dist = np.square(dist)

        for grid_point_i in range(0, dist.shape[0]): # go over each coordinate in a new grid and interpolate the features
            x_new, y_new = new_coord_patch[grid_point_i] # coordinate in a new grid
            r_tmp = np.sqrt(x_new ** 2 + y_new ** 2) # length of the radius from the center to the new point

            # Because our grid has negative coordinates, we will shift then to have only positive coordinates:
            column_i = x_new + radius # row index of final patch grid
            row_i = - y_new + radius -1 # column index of final patch grid

            # If the point outside of patch -
            if r_tmp>radius:
                patch_grid[row_i][column_i][feature_i] = 0
                continue

            if dist[grid_point_i][0]==0: # if the coordinate is for the neighbor that doesn't exist
                neigh_index_i = indx_old[grid_point_i][0]
                if x_new == 0 and y_new == 0:
                    patch_grid[row_i][column_i][feature_i] = input_feat[0][feature_i] # if center point
                else:
                    patch_grid[row_i][column_i][feature_i] = input_feat[neigh_index_i][feature_i]
                continue

            dist_grid_point = dist[grid_point_i]
            result_grid_points = indx_old[grid_point_i] #points to interpolate
            dist_to_include = []
            result_to_include = [] # old index list
            # Several old coordinates can map to a one new grid coordinate. Remove the redundancy:
            for i, result_i in enumerate(result_grid_points):
                if result_i not in result_to_include:
                    result_to_include.append(result_i)
                    dist_to_include.append(dist_grid_point[i])

            if interpolate:
                total_dist = np.sum(1 / np.array(dist_to_include))
                interpolated_value = 0
                for i, result_old_i in enumerate(result_to_include):
                        interpolated_value += input_feat[result_old_i][feature_i] * (1/ dist_to_include[i])/total_dist
                patch_grid[row_i][column_i][feature_i] = interpolated_value
            else:
                try:
                    patch_grid[row_i][column_i][feature_i] = input_feat[result_grid_points[0]][feature_i]
                except IndexError:
                    patch_grid[row_i][column_i][feature_i] = 0

    # print("Performing {} random rotations...".format(n_rotations))
    # for patch_i in range(0, x.shape[0]):
    #     random_rotations = np.random.randint(low=1, high=360, size=n_rotations-1)
    #     for i, angle in enumerate(random_rotations):
    #         #print('Rotating on angle {}'.format(angle))
    #         for feature_i in range(0, patch_grid.shape[1]):
    #             patch_grid[patch_i][feature_i][i+1] = ndimage.rotate(patch_grid[patch_i][feature_i][0], angle, reshape=False)
    return patch_grid

def read_patch(pid, ch, config):
    patch_dir = config['dirs']['patches'] + pid + '/'

    rho = np.load(patch_dir + '{}_{}_rho_wrt_center.npy'.format(pid, ch), allow_pickle=True)
    theta = np.load(patch_dir + '{}_{}_theta_wrt_center.npy'.format(pid, ch), allow_pickle=True)
    input_feat = np.load(patch_dir + '{}_{}_input_feat.npy'.format(pid, ch), allow_pickle=True)
    resnames = np.load(patch_dir + '{}_{}_resnames.npy'.format(pid, ch), allow_pickle=True)
    resnames = np.expand_dims(resnames, axis=1)

    # ## read interaction features
    # all_interact_feat = []
    # if 'interact_feat' in config.keys():
    #     for interact_feat in config['interact_feat'].keys():
    #         if config['interact_feat'][interact_feat] == True:
    #             feat =  np.load(patch_dir + '{}_{}.npy'.format(pid, interact_feat), allow_pickle=True)
    #             all_interact_feat.append(feat)
    #             ## read interaction features
    #
    #     all_interact_feat = np.concatenate(all_interact_feat, axis=-1)

    # Read 3D coordinates
    coord_3d = np.load(patch_dir + '{}_{}_coordinates.npy'.format(pid, ch), allow_pickle=True)

    return rho, theta, input_feat, resnames, coord_3d

def remove_comments(pdb_path, pdb_tmp_path):
    """
    Standartize PDB file by adding white spaces and making each line exactly 80 characters
    :param pdb_path: input PDB
    :param pdb_tmp_path: output PDB with fixed format
    :return: None
    """
    with open(pdb_path, 'r') as in_pdb:
        with open(pdb_tmp_path, 'w') as out:
            for line in in_pdb.readlines():
                if "USER" not in line:
                    newline = []
                    for i in range(80):
                        if i<len(line.strip('\n')):
                            newline.append(line[i])
                        else:
                            newline.append(' ')
                    if line[:4]=="ATOM" or line[:6]=="HETATM":
                        newline[77]=newline[13]
                        #pdb.set_trace()
                    out.write(''.join(newline)+'\n')
                    #out.write(line)
    return None


def compute_dssp(ppi, config):
    # Compute DSSP values as described in https://biopython.org/docs/1.75/api/Bio.PDB.DSSP.html

    pid, ch1, ch2 = ppi.split('_')

    # raw_pdb_dir = config['dirs']['raw_pdb']
    tmp_dir = config['dirs']['tmp']

    # pdb_path = tmp_dir+'{}_{}_{}.pdb'.format(pid,ch1,ch2)
    # extractPDB(raw_pdb_dir+"pdb{}.ent".format(pid.lower()), pdb_path, chain_ids=ch1+ch2)
    pdb_path = config['dirs']['protonated_pdb'] + '{}.pdb'.format(pid)
    pdb_tmp_path = f"{tmp_dir}/{pid}.pdb"

    # remove hydrogens
    remove_comments(pdb_path, pdb_tmp_path)

    parser = PDBParser(QUIET=1)
    struct = parser.get_structure(pid, pdb_tmp_path)

    model = struct[0]

    dssp = DSSP(model, pdb_tmp_path, dssp='mkdssp')  # example of a key: ('A', (' ', 1147, ' '))

    # Remove temporary file
    os.remove(pdb_tmp_path)
    return dssp

def convert_dssp_to_feat(dssp, names_grid):
    """
    Convert DSSP object into grid of features

    Hydrogen bonds for each chain will be computed separate,
                    as residue from one side can form bonds with multiple residues from the other side.

    PHI PSI - IUPAC peptide backbone torsion angles
    :param dssp:
    :param names_grid:
    :return: numpy array dssp_features
    0 - Relative ASA;
    1 - NH–>O_1_relidx
    2 -

    """

    # 0 - Relative ASA | dssp[3]

    # 1 - NH–>O_1_energy | dssp[7]
    # 2 - O–>NH_1_energy | dssp[9]
    # 3 - NH–>O_2_energy | dssp[11]
    # 4 - O–>NH_2_energy | dssp[13]

    # One hot variables:
    # 5 - alpha helix (4-12, 3-10. or Pi helix) | dssp[2] in ['H', 'G', 'I']
    # 6 - beta sheet (Isolated beta-bridge) | dssp[2] == 'B'
    # 7 - strand | dssp[2] == 'E'
    # 8 - turn or bend | dssp[2] is in ['T', 'S']

    dssp_features = np.zeros((names_grid.shape[0], names_grid.shape[1], 1))

    for i in range(names_grid.shape[0]):
        for j in range(names_grid.shape[1]):
            curr_name = names_grid[i][j][0] # example A:107:HIS-1621:CD2
            if curr_name!=0:
                # Read the residue from the array of names of a patch pair
                fields = curr_name.split(':')
                chain, resid = fields[0], fields[1]

                # Construct a key based on the current residue from two proteins
                # key example: ('A', (' ', 219, ' '))
                for key_i in dssp.keys():
                    if key_i[0]==chain and key_i[1][1] == int(resid):
                        dssp_key =key_i

                try:
                    dssp_features_i = dssp[dssp_key]
                except:
                    dssp_features[i][j][0] = 0
                    continue

                # Relative ASA:
                try:
                    dssp_features[i][j][0] = dssp_features_i[3]
                except:
                    dssp_features[i][j][0] = 0

                # # H-bond energies:
                # dssp_features[i][j][1] = - dssp_features_i[7]
                # dssp_features[i][j][2] = - dssp_features_i[9]
                # dssp_features[i][j][3] = - dssp_features_i[11]
                # dssp_features[i][j][4] = - dssp_features_i[13]
                # # Secondary structure:
                # dssp_features[i][j][5] = 1 if dssp_features_i[2] in ['H', 'G', 'I'] else 0
                # dssp_features[i][j][6] = 1 if dssp_features_i[2]=='B' else 0
                # dssp_features[i][j][7] = 1 if dssp_features_i[2]=='E' else 0
                # dssp_features[i][j][8] = 1 if dssp_features_i[2] in ['T', 'S'] else 0

    return dssp_features

# def find_optimal_rotation(p1_coord_grid, p2_coord_grid):
#     optimal_dist_grid = np.sqrt(np.sum(np.square(p1_coord_grid - p2_coord_grid), axis=-1))
#     optimal_rotation = 0
#
#     for angle_i in range(360):
#         p2_rotated = ndimage.rotate(p2_coord_grid, angle_i, reshape=False, order=0)
#         dist_grid = np.sqrt(np.sum(np.square(p1_coord_grid - p2_rotated), axis=-1))
#         if dist_grid.mean() < optimal_dist_grid.mean():
#             optimal_rotation=angle_i
#             optimal_dist_grid = dist_grid
#     return optimal_rotation, optimal_dist_grid
def find_optimal_rotation(p1_rho, p1_theta, p2_rho, p2_theta, p1_coord_3d, p2_coord_3d, radius):
    # return (p1target_x, p1target_y), (p2_x, p2_y), where (p2_x, p2_y)

    optimal_angle = 0
    optimal_distance = np.inf
    angle_step = 6.28/100 # make 100 rotations
    curr_angle=0
    p1target_x, p1target_y = polar_to_cartesian(p1_rho, p1_theta)
    p1_coord_grid = compute_patch_grid(p1target_x, p1target_y, p1_coord_3d, radius)

    while curr_angle<6.28:
        p2_x, p2_y = polar_to_cartesian(p2_rho, p2_theta, curr_angle) # rotate only p2
        p2_coord_grid = compute_patch_grid(p2_x, p2_y, p2_coord_3d, radius)
        dist_grid = np.sqrt(np.sum(np.square(p1_coord_grid - p2_coord_grid), axis=-1))
        avg_dist = dist_grid.mean()
        if avg_dist < optimal_distance:
            optimal_angle=curr_angle
            optimal_distance = avg_dist

        curr_angle+=angle_step
    print(f"Optimal angle: {optimal_angle} radians.")
    # print(f"Average distance: {optimal_distance}")
    p2_x, p2_y = polar_to_cartesian(p2_rho, p2_theta, optimal_angle)  # rotate only p2
    return (p1target_x, p1target_y), (p2_x, p2_y)

def convert_one_patch(ppi, config):
    pid, ch1, ch2 = ppi.split('_')

    out_grid = config['dirs']['grid'] + '{}_{}_{}.npy'.format(pid, ch1, ch2)
    out_resnames = config['dirs']['grid'] + '{}_{}_{}_resnames.npy'.format(pid, ch1, ch2)

    radius = config['ppi_const']['patch_r']

    p1_rho, p1_theta, p1_input_feat, p1_resnames, p1_coord_3d = read_patch(pid, ch1, config)
    p2_rho, p2_theta, p2_input_feat, p2_resnames, p2_coord_3d = read_patch(pid, ch2, config)

    ##### Compute optimal grid
    ## Incrementally increase theta of all coordinates and find the best match
    # (p1target_x, p1target_y), (p2_x, p2_y) = find_optimal_rotation(p1_rho, p1_theta, p2_rho, p2_theta, p1_coord_3d, p2_coord_3d, radius)
    p1target_x, p1target_y = polar_to_cartesian(p1_rho, p1_theta)
    p2_x, p2_y = polar_to_cartesian(p2_rho, p2_theta)

    p1target_patch_grid = compute_patch_grid(p1target_x, p1target_y, p1_input_feat, radius)  # (r, r, n_feat)
    p2_patch_grid = compute_patch_grid(p2_x, p2_y, p2_input_feat, radius)  # (r, r, n_feat)

    p1name_grid = compute_patch_grid(p1target_x, p1target_y, p1_resnames, radius, interpolate=False, stringarray=True)
    p2name_grid = compute_patch_grid(p2_x, p2_y, p2_resnames, radius, interpolate=False, stringarray=True)

    # if 'interact_feat' in config.keys():
    #     if config['interact_feat']['atom_dist']==True:
    #         # Interpolate coordinates to the patch, so that we can compute a grid with atom distances
    p1_coord_grid = compute_patch_grid(p1target_x, p1target_y, p1_coord_3d, radius)
    p2_coord_grid = compute_patch_grid(p2_x, p2_y, p2_coord_3d, radius)
    dist_grid = np.sqrt(np.sum(np.square(p1_coord_grid - p2_coord_grid), axis=-1))
    print(f"Average distance between surfaces: {dist_grid.mean()}")

    single_grid = np.concatenate([p1target_patch_grid, p2_patch_grid, np.expand_dims(dist_grid, axis=-1)], axis=-1)
    names_grid = np.concatenate([p1name_grid, p2name_grid], axis=-1)

    # if config['interact_feat']['dssp']==True:
    dssp = compute_dssp(ppi, config)
    dssp_grid_1 = convert_dssp_to_feat(dssp, p1name_grid)
    dssp_grid_2 = convert_dssp_to_feat(dssp, p2name_grid)
    single_grid = np.concatenate([single_grid, dssp_grid_1, dssp_grid_2], axis=-1)


    # if len(all_interact_feat)>0:
    #     interact_patch_grid = compute_patch_grid(p1target_x, p1target_y, all_interact_feat, radius)
    #     single_grid = np.concatenate([single_grid, interact_patch_grid], axis=-1)

    np.save(out_grid, single_grid)
    np.save(out_resnames, names_grid)

    return None

def convert_to_images(ppi_list, config):
    for ppi in tqdm(ppi_list):
        pid, ch1, ch2 = ppi.split('_')
        out_grid = config['dirs']['grid'] + '{}_{}_{}.npy'.format(pid, ch1, ch2)
        print("Computing image for {}".format(ppi))
        if os.path.exists(out_grid):
            print("Image is already computed {}. Skipping".format(ppi))
            continue
        convert_one_patch(ppi, config)
    return None