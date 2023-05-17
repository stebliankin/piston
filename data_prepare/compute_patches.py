import pymesh
import pdb
import time
import numpy as np

from sklearn.manifold import MDS
from masif.source.masif_modules.read_data_from_surface import compute_ddc, normalize_electrostatics
from masif.source.geometry.compute_polar_coordinates import call_mds, compute_thetas, dict_to_sparse, compute_theta_all_fast

from sklearn.neighbors import KDTree
import os
from tqdm import tqdm
import scipy
import networkx as nx

from utils.utils import get_date
# def compute_patch_chain(pid, ch, config):
#
#
#
#
#     return

def compute_patch_center(mesh1, mesh2, radius):

    # Compute patch center and select all verticies within the range "radius"

    iface_vert1 = get_iface_verticies(mesh1)
    iface_vert2 = get_iface_verticies(mesh2)

    iface_vert_all = np.concatenate((iface_vert1, iface_vert2), axis=0)

    # Compute geometric center
    center_point = np.mean(iface_vert_all, axis=0)

    kdt1 = KDTree(mesh1.vertices)
    #pdb.set_trace()
    d, indx_cent1 = kdt1.query(np.expand_dims(center_point, axis=0))
   # patch1_indx = kdt1.query_radius(np.expand_dims(center_point, axis=0), r=radius)


    kdt2 = KDTree(mesh2.vertices)
    d, indx_cent2 = kdt2.query(np.expand_dims(center_point, axis=0))
  #  patch2_indx = kdt2.query_radius(np.expand_dims(center_point, axis=0), r=radius)

    return center_point, indx_cent1[0][0], indx_cent2[0][0] #, patch1_indx[0], patch2_indx[0]

def get_iface_verticies(mesh):
    iface = mesh.get_attribute('vertex_iface')
    vertices = mesh.vertices
    iface_indx = np.where(iface>0)
    #pdb.set_trace()
    if len(iface_indx[0])==0:
        print('WARNING:: No interface found!')
        iface_indx = np.where(iface==0)
    return vertices[iface_indx]


def compute_theta_all(D, vertices, faces, normals, idx, radius, patch_center_i):
    # Reference: https://github.com/LPDI-EPFL/masif/blob/2a370518e0d0d0b0d6f153f2f10f6630ae91f149/source/geometry/compute_polar_coordinates.py#L300

    mymds = MDS(n_components=2, n_init=1, max_iter=50, dissimilarity='precomputed', n_jobs=10)
    all_theta = []
    i = patch_center_i
    if i % 100 == 0:
        print(i)
    # Get the pairs of geodesic distances.

    neigh = D[i].nonzero()
    ii = np.where(D[i][neigh] < radius)[1]
    neigh_i = neigh[1][ii]
    pair_dist_i = D[neigh_i, :][:, neigh_i]
    pair_dist_i = pair_dist_i.todense()

    # Plane_i: the 2D plane for all neighbors of i
    plane_i = call_mds(mymds, pair_dist_i)

    # Compute the angles on the plane.
    theta = compute_thetas(plane_i, i, vertices, faces, normals, neigh_i, idx)
    return theta


def compute_polar_coordinates(mesh, patch_center_i,  radius=12, max_vertices=200):
    """
    # Reference: https://github.com/LPDI-EPFL/masif/blob/2a370518e0d0d0b0d6f153f2f10f6630ae91f149/source/geometry/compute_polar_coordinates.py#L19
    compute_polar_coordinates: compute the polar coordinates for every patch in the mesh.
    Returns:
        rho: radial coordinates for each patch. padded to zero.
        theta: angle values for each patch. padded to zero.
        neigh_indices: indices of members of each patch.
        mask: the mask for rho and theta
    """

    # Vertices, faces and normals
    vertices = mesh.vertices
    faces = mesh.faces
    norm1 = mesh.get_attribute('vertex_nx')
    norm2 = mesh.get_attribute('vertex_ny')
    norm3 = mesh.get_attribute('vertex_nz')
    normals = np.vstack([norm1, norm2, norm3]).T

    # Graph
    G = nx.Graph()
    n = len(mesh.vertices)
    G.add_nodes_from(np.arange(n))

    # Get edges
    f = np.array(mesh.faces, dtype=int)
    rowi = np.concatenate([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]], axis=0)
    rowj = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]], axis=0)
    edges = np.stack([rowi, rowj]).T
    verts = mesh.vertices

    # Get weights
    edgew = verts[rowi] - verts[rowj]
    edgew = scipy.linalg.norm(edgew, axis=1)
    wedges = np.stack([rowi, rowj, edgew]).T

    G.add_weighted_edges_from(wedges)
    start = time.clock()

    dists = nx.all_pairs_dijkstra_path_length(G, cutoff=radius * 2)

    d2 = {}
    for key_tuple in dists:
        d2[key_tuple[0]] = key_tuple[1]
    end = time.clock()
    print('Dijkstra took {:.2f}s'.format((end - start)))
    D = dict_to_sparse(d2)

    # Compute the faces per vertex.
    idx = {}
    for ix, face in enumerate(mesh.faces):
        for i in range(3):
            if face[i] not in idx:
                idx[face[i]] = []
            idx[face[i]].append(ix)

    i = np.arange(D.shape[0])
    # Set diagonal elements to a very small value greater than zero..
    D[i, i] = 1e-8
    # Call MDS for all points.
    mds_start_t = time.clock()

    theta = compute_theta_all(D, vertices, faces, normals, idx, radius, patch_center_i)

    # Output a few patches for debugging purposes.
    # extract a patch
    # for i in [0,100,500,1000,1500,2000]:
    #    neigh = D[i].nonzero()
    #    ii = np.where(D[i][neigh] < radius)[1]
    #    neigh_i = neigh[1][ii]
    #    subv, subn, subf = extract_patch(mesh, neigh_i, i)
    #    # Output the patch's rho and theta coords
    #    output_patch_coords(subv, subf, subn, i, neigh_i, theta[i], D[i, :])

    mds_end_t = time.clock()
    print('MDS took {:.2f}s'.format((mds_end_t - mds_start_t)))

    n = len(d2)
    theta_out = np.zeros((max_vertices))
    rho_out = np.zeros((max_vertices))
    mask_out = np.zeros((max_vertices))

    i = patch_center_i
    # Assemble output.

    dists_i = d2[i]
    sorted_dists_i = sorted(dists_i.items(), key=lambda kv: kv[1])
    neigh = [int(x[0]) for x in sorted_dists_i[0:max_vertices]]
    rho_out[:len(neigh)] = np.squeeze(np.asarray(D[i, neigh].todense()))
    theta_out[:len(neigh)] = np.squeeze(theta[neigh])
    mask_out[:len(neigh)] = 1
    # have the angles between 0 and 2*pi
    theta_out[theta_out < 0] += 2 * np.pi

    return rho_out, theta_out, neigh, mask_out

def read_data_from_surface(ply_fn, patch_center_i, config):
    """
    # Reference:
    #   https://github.com/LPDI-EPFL/masif/blob/2a370518e0d0d0b0d6f153f2f10f6630ae91f149/source/masif_modules/read_data_from_surface.py#L14
    # Read data from a ply file -- decompose into patches.
    # Returns:
    # list_desc: List of features per patch
    # list_coords: list of angular and polar coordinates.
    # list_indices: list of indices of neighbors in the patch.
    # list_sc_labels: list of shape complementarity labels (computed here).
    """
    mesh = pymesh.load_mesh(ply_fn)

    # Normals:
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1, n2, n3], axis=1)

    # Compute the angular and radial coordinates.
    radius = config['ppi_const']['patch_r']
    points_in_patch = config['ppi_const']['points_in_patch']
    rho, theta, neigh_indices, mask = compute_polar_coordinates(mesh, patch_center_i, radius=radius,
                                                                max_vertices=points_in_patch)

    # Compute the principal curvature components for the shape index.
    mesh.add_attribute("vertex_mean_curvature")
    H = mesh.get_attribute("vertex_mean_curvature")
    mesh.add_attribute("vertex_gaussian_curvature")
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index
    si = (k1 + k2) / (k1 - k2)
    si = np.arctan(si) * (2 / np.pi)

    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)

    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")

    # Hydropathy features
    # Normalize hydropathy by dividing by 4.5
    hphob = mesh.get_attribute("vertex_hphob") / 4.5

    # Iface labels (for ground truth only)
    if "vertex_iface" in mesh.get_attribute_names():
        iface_labels = mesh.get_attribute("vertex_iface")
    else:
        iface_labels = np.zeros_like(hphob)

    # n: number of patches, equal to the number of vertices.
    n = len(mesh.vertices)

    input_feat = np.zeros((points_in_patch, 5))

    # Compute the input features for each patch.
    vix = patch_center_i
    # Patch members.
    neigh_vix = np.array(neigh_indices)

    # Compute the distance-dependent curvature for all neighbors of the patch.
    patch_v = mesh.vertices[neigh_vix]
    patch_n = normals[neigh_vix]
    patch_cp = np.where(neigh_vix == vix)[0][0]  # central point
    mask_pos = np.where(mask == 1.0)[0]  # nonzero elements
    patch_rho = rho[mask_pos]  # nonzero elements of rho
    ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)

    # impute missing shape indicies with mean value for the whole patch
    si_patch = si[neigh_vix]

    si_patch = np.nan_to_num(si_patch, nan=np.nanmean(si_patch)) # replace nan values

    input_feat[:len(neigh_vix), 0] = si_patch
    input_feat[:len(neigh_vix), 1] = ddc
    input_feat[:len(neigh_vix), 2] = hbond[neigh_vix]
    input_feat[:len(neigh_vix), 3] = charge[neigh_vix]
    input_feat[:len(neigh_vix), 4] = hphob[neigh_vix]

    return input_feat, rho, theta, mask, neigh_indices, iface_labels, np.copy(mesh.vertices)


# def read_ply(pid, ch, config):
#     ply_file = config['dirs']['surface_ply'] + pid + '_' + ch + '.ply'
#     ply_header = []
#     ply_entries = []
#     header=True
#     with open(ply_file, 'r') as f:
#         for line in f.readlines():
#             if header:
#                 ply_header.append(line)
#             else:
#                 ply_entries.append(line)
#             if "end_header" in line:
#                 header=False
#     return ply_header, ply_entries


# def crop_ply_patch(pid, ch, patch_indx, config):
#
#     out_cropped_ply_patch = config['dirs']['patch_ply'] + pid+'_'+ch + '_patch.ply'
#
#     ply_header, ply_entries = read_ply(pid,ch, config)
#     ply_entries = np.array(ply_entries)
#     ply_entries = ply_entries[patch_indx]
#
#     # write cropped ply
#     with open(out_cropped_ply_patch, 'w') as out:
#         for header_line in ply_header:
#             out.write(header_line)
#         for entry_line in ply_entries:
#             out.write(entry_line)


def save_precompute(pid, ch, config, input_feat, rho, theta, mask, neigh_indices, iface_labels, verts, center_patch_i, patch_coord):

    out_patch_dir = config['dirs']['patches']
    my_precomp_dir = out_patch_dir + pid + '/'
    if not os.path.exists(my_precomp_dir):
        os.mkdir(my_precomp_dir)

    np.save(my_precomp_dir + pid + '_' + ch + '_rho_wrt_center', rho)
    np.save(my_precomp_dir + pid + '_' + ch + '_theta_wrt_center', theta)
    np.save(my_precomp_dir + pid + '_' + ch + '_input_feat', input_feat)
    np.save(my_precomp_dir + pid + '_' + ch + '_mask', mask)
    np.save(my_precomp_dir + pid + '_' + ch + '_list_indices', neigh_indices)
    np.save(my_precomp_dir + pid + '_' + ch + '_iface_labels', iface_labels)
    # Save x, y, z
    np.save(my_precomp_dir + pid + '_' + ch + '_X.npy', verts[center_patch_i, 0])
    np.save(my_precomp_dir + pid + '_' + ch + '_Y.npy', verts[center_patch_i, 1])
    np.save(my_precomp_dir + pid + '_' + ch + '_Z.npy', verts[center_patch_i, 2])

    np.save(my_precomp_dir + pid + '_' + ch + '_X_all.npy', verts[:, 0])
    np.save(my_precomp_dir + pid + '_' + ch + '_Y_all.npy', verts[:, 1])
    np.save(my_precomp_dir + pid + '_' + ch + '_Z_all.npy', verts[:, 2])

    np.save(my_precomp_dir + pid + '_' + ch + '_coordinates.npy', patch_coord)


# def save_precompute_interaction(pid, inter_feat, config):
#     """
#     Save interaction features into numpy files
#     :param pid: PDB ID
#     :param inter_feat: dictionary of interaction features
#     :param config: configuration dictionary
#     :return: None
#     """
#     out_patch_dir = config['dirs']['patches']
#     my_precomp_dir = out_patch_dir + pid + '/'
#
#     if not os.path.exists(my_precomp_dir):
#         os.mkdir(my_precomp_dir)
#
#     for key in inter_feat.keys():
#         np.save(my_precomp_dir + pid + '_{}.npy'.format(key), inter_feat[key])
#
#     return None

def compute_patches(ppi_list, config, overwrite=False):
    radius = config['ppi_const']['patch_r']

    print("**** [ {} ] Compute patches".format(get_date()))
    print("{}".format(ppi_list))

    for ppi in tqdm(ppi_list):
        print("Computing patch pair for {}".format(ppi))
        pid, ch1, ch2 = ppi.split('_')

        out_feat1 = config['dirs']['patches']+'{}/{}_{}_input_feat.npy'.format(pid, pid, ch1)
        out_feat2 = config['dirs']['patches']+'{}/{}_{}_input_feat.npy'.format(pid, pid, ch2)
        if os.path.exists(out_feat1) and os.path.exists(out_feat2) and not overwrite:
            print('Patch already computed for {}. Skipping...'.format(ppi))
            continue

        ply_dir = config['dirs']['surface_ply']

        ply_fn1 = ply_dir + '{}_{}.ply'.format(pid, ch1)
        ply_fn2 = ply_dir + '{}_{}.ply'.format(pid, ch2)

        mesh1 = pymesh.load_mesh(ply_fn1)
        mesh2 = pymesh.load_mesh(ply_fn2)

        # Compute patch center
        patch_center, indx_c1, indx_c2 = compute_patch_center(mesh1, mesh2, radius)
        print('Center of the interaction: {}'.format(patch_center))
        # crop_ply_patch(pid, ch1, patch1_indx, config)
        # crop_ply_patch(pid, ch2, patch2_indx, config)

        input_feat1, rho1, theta1, mask1, neigh_indices1, iface_labels1, verts1 = read_data_from_surface(ply_fn1, indx_c1, config)
        input_feat2, rho2, theta2, mask2, neigh_indices2, iface_labels2, verts2 = read_data_from_surface(ply_fn2, indx_c2, config)

        # Compute 3D coordinates of the patch (used to compute the distance between atoms)
        points_in_patch = config['ppi_const']['points_in_patch']
        patch_coord1, patch_coord2 = np.zeros((points_in_patch, 3)), np.zeros((points_in_patch, 3))
        patch_coord1[:len(neigh_indices1)] = verts1[neigh_indices1]
        patch_coord2[:len(neigh_indices2)] = verts2[neigh_indices2]

        # print("Computing interaction features...")
        # inter_feat = {}
        # if 'interact_feat' in config.keys():
        #     ## Distance between atoms
        #     if 'atom_dist' in config['interact_feat'].keys() and config['interact_feat']['atom_dist']:
        #         # X1, Y1, Z1 = verts1[indx_c1, 0], verts1[indx_c1, 1], verts1[indx_c1, 2]
        #         # X2, Y2, Z2 = verts2[indx_c2, 0], verts2[indx_c2, 1], verts2[indx_c2, 2]
        #         ## Get coordinates of the patch
        #         patch_coord1 = verts1[neigh_indices1]
        #         patch_coord2 = verts2[neigh_indices2]
        #
        #         inter_feat['atom_dist'] = compute_atom_dist(patch_coord1, patch_coord2)
        #         # # reverse to ensure that two distances are the same
        #         # inter_feat['atom_dist2'] = compute_atom_dist(patch_coord2, patch_coord1)
        #
        #     # save interaction features
        #     save_precompute_interaction(pid, inter_feat, config)

        save_precompute(pid, ch1, config, input_feat1, rho1, theta1, mask1, neigh_indices1, iface_labels1, verts1, indx_c1, patch_coord1)
        save_precompute(pid, ch2, config, input_feat2, rho2, theta2, mask2, neigh_indices2, iface_labels2, verts2, indx_c2, patch_coord2)




# def compute_atom_dist(patch_coord1, patch_coord2):
#     # For each coordinate find the closest atom from the other side
#
#     kdt1 = KDTree(patch_coord1)
#     dist, _ = kdt1.query(patch_coord2)
#
#     return dist

# def compute_interaction_features(config):
#     # Compute distance between atoms:
#     if 'features' in config.keys() and 'atom_dist' in config['features'].keys() and config['features']['atom_dist']:
#         atom_dist = compute_atom_dist()

    # Compute