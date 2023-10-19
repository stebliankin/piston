import shutil

import pymesh #Importing pymesh here avoids library conflict (CXXABI_1.3.11)
from Bio.PDB import *
from subprocess import Popen, PIPE
from masif.source.input_output.protonate import protonate
import os
import pdb
from scipy.spatial import cKDTree
import numpy as np
import time
from utils.utils import get_date, extract_pdb_chain
from tqdm import tqdm

def protonate_pdb(ppi, config):
    """
    downlaod and add hydrogens to PPI
    """
    pid = ppi.split('_')[0]

    # Download pdb
    pdb_filename = config['dirs']['raw_pdb'] + pid + '.pdb'
    if not os.path.exists(pdb_filename):
        pdbl = PDBList()
        pdb_filename = pdbl.retrieve_pdb_file(pid, pdir=config['dirs']['raw_pdb'], file_format='pdb')
    else:
        ## Remove MODEL line
        tmp_filename = config['dirs']['raw_pdb'] + pid + '_tmp.pdb'
        os.rename(pdb_filename, tmp_filename)
        with open(pdb_filename, 'w') as out:
            with open(tmp_filename, 'r') as f:
                for line in f:
                    if "MODEL" not in line:
                        out.write(line)
        os.remove(tmp_filename)

    # Protonate downloaded file
    protonated_file = config['dirs']['protonated_pdb']+"/"+pid+".pdb"
    protonate(pdb_filename, protonated_file)

def download(ppi_list, config, to_write=None):
    start = time.time()
    print("**** [ {} ] Start Downloading PDBs...".format(get_date()))
    print(ppi_list)

    processed_ppi = []
    for i in tqdm(range(len(ppi_list))):
        ppi = ppi_list[i]
        pid = ppi.split('_')[0]

        raw_pdb_filename = config['dirs']['protonated_pdb']+"/"+pid+".pdb"

        if not os.path.exists(raw_pdb_filename):
            protonate_pdb(ppi, config)
        else:
            print("PDB file {} already exists. Skipping...".format(pid))

        if os.path.exists(raw_pdb_filename):
            processed_ppi.append(ppi)

    if to_write is not None:
        with open(to_write, 'w') as out:
            for ppi in processed_ppi:
                out.write(ppi+'\n')

    print("**** [ {} ] Done with downloading PDBs.".format(get_date()))
    print("**** [ {} ] Took {:.2f}min.".format(get_date(), (time.time()-start)/60))
    return processed_ppi


def select_single_model(pdb_path, pdb_path_updated):
    with open(pdb_path_updated, 'w') as out:
        with open(pdb_path, 'r') as f:
            for line in f.readlines():
                if line[:5]=="MODEL" or line[:6]=="REMARK":
                    pass
                elif line[:6]=="ENDMDL":
                    break
                else:
                    out.write(line)

def get_coord_dict(pid, pdb_path, chain):

    parser = PDBParser(QUIET=True)
    #pdb.set_trace()
    try:
        pdb_struct = parser.get_structure(pid, pdb_path)
    except ValueError: # tbe PDB file contain multiple models
        pdb_path_updated = pdb_path.replace('.pdb','') + '_singleModel.pdb'
        select_single_model(pdb_path, pdb_path_updated)
        pdb_struct = parser.get_structure(pid, pdb_path_updated)

    RES_dict = {'atom_id': [], 'res_id': [], 'chain_id': [], 'atom_coord': []}

    all_atom_res_chain_pairs = []

    for i, atom in enumerate(pdb_struct.get_atoms()):
        # atom_name = atom.name
        # res_name = atom.parent.resname
        res_id = atom.parent.id[1]
        chain_id = atom.get_parent().get_parent().get_id()
        atom_coord = list(atom.get_coord())
        atom_id = atom.serial_number

        if chain_id in chain:
            # The condition below will make sure that if PDB has multiple models, only the first one will be included.
            if (atom_id, res_id, chain_id) not in all_atom_res_chain_pairs:
                all_atom_res_chain_pairs.append((atom_id, res_id, chain_id))
                RES_dict['atom_id'].append(atom_id)
                RES_dict['res_id'].append(res_id)
                RES_dict['chain_id'].append(chain_id)
                RES_dict['atom_coord'].append(atom_coord)
    return RES_dict

def crop_pdb_one(ppi, config, use_refined=True):
    pid, ch1, ch2 = ppi.split('_')

    if use_refined:
        pdb_file = f"{config['dirs']['refined']}/{ppi}/refined-out-{ppi}_1.ref.pdb"  # config['dirs']['refined'] + pid + '.pdb'
    if not use_refined or not os.path.exists(pdb_file):
        print(f"Loading original PDB...")
        pdb_file = f"{config['dirs']['protonated_pdb']}/{pid}.pdb"

    crop_r = config['ppi_const']['crop_r']
    contact_d = config['ppi_const']['contact_d']
    out_file = config['dirs']['cropped_pdb'] + pid + '.pdb'

    if os.path.exists(out_file) and use_refined:
        # Skip if file already exists
        print("Cropped PDB already exists. Skipping")
        return

    res_dict_1 = get_coord_dict(pid, pdb_file, ch1)
    res_dict_2 = get_coord_dict(pid, pdb_file, ch2)

    if len(res_dict_1['res_id']) == 0 and len(res_dict_2['res_id']) == 0:
        # The PDB file is empty. Skpping...
        print("ERROR::PDB file is empty")
        return

    # Search for contact points within
    pdb_tree_1 = cKDTree(res_dict_1['atom_coord'])

    all_dist, all_idx_1 = pdb_tree_1.query(res_dict_2[
                                               'atom_coord'])  # all_idx_1 - index of the first array; len(all_indx_1) is the lenght if the second array

    contact_indx2 = np.where(all_dist < contact_d)
    contact_indx1 = np.unique(all_idx_1[contact_indx2])

    center_iface1 = np.mean(np.array(res_dict_1['atom_coord'])[contact_indx1], axis=0)
    center_iface2 = np.mean(np.array(res_dict_2['atom_coord'])[contact_indx2], axis=0)

    # Compute all residues to include
    dist_func1 = lambda x: (center_iface1[0] - x[0]) ** 2 + (center_iface1[1] - x[1]) ** 2 + (
                center_iface1[2] - x[2]) ** 2
    dist_func2 = lambda x: (center_iface2[0] - x[0]) ** 2 + (center_iface2[1] - x[1]) ** 2 + (
                center_iface2[2] - x[2]) ** 2

    dist_to_center1 = np.array([dist_func1(xi) for xi in res_dict_1['atom_coord']])
    dist_to_center2 = np.array([dist_func2(xi) for xi in res_dict_2['atom_coord']])

    min_dist1 = min(
        dist_to_center1)  # if the center is outside of the surface, upper bound distance is crop_r + min_dist
    min_dist2 = min(dist_to_center2)

    res_to_include1_indx = np.where(dist_to_center1 < (crop_r ** 2 + min_dist1 ** 2))
    res_to_include2_indx = np.where(dist_to_center2 < (crop_r ** 2 + min_dist2 ** 2))

    res_to_include1 = [res_dict_1['chain_id'][i] + ':' + str(res_dict_1['res_id'][i]) for i in
                       range(0, len(res_dict_1['chain_id']))]
    res_to_include1 = np.unique(np.array(res_to_include1)[res_to_include1_indx])

    res_to_include2 = [res_dict_2['chain_id'][i] + ':' + str(res_dict_2['res_id'][i]) for i in
                       range(0, len(res_dict_2['chain_id']))]
    res_to_include2 = np.unique(np.array(res_to_include2)[res_to_include2_indx])

    res_to_include = np.append(res_to_include1, res_to_include2)

    with open(out_file, 'w') as out:
        with open(pdb_file, 'r') as f:
            for line in f.readlines():
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    ch = line[21]
                    res_id = int(line[22:26])

                    if '{}:{}'.format(ch, res_id) in res_to_include:
                        out.write(line)
                else:
                    out.write(line)

    extract_pdb_chain(out_file, config['dirs']['cropped_pdb'] + '/{}_{}.pdb'.format(pid, ch1), ch1)
    extract_pdb_chain(out_file, config['dirs']['cropped_pdb'] + '/{}_{}.pdb'.format(pid, ch2), ch2)

def crop_pdb(ppi_list, config):
    print("**** [ {} ] Cropping complexes to {}A radius from interaction center...".format(get_date(),
                                                                                   config['ppi_const']['crop_r']))
    processed_ppi = []
    for ppi in tqdm(ppi_list):
        pid, ch1, ch2 = ppi.split('_')

        pdb_file = f"{config['dirs']['refined']}/{ppi}/refined-out-{ppi}_1.ref.pdb"#config['dirs']['refined'] + pid + '.pdb'
        if not os.path.exists(pdb_file):
            print(f"WARNING::Refined {ppi} doesn't exist. Loading original PDB...")
            pdb_file = f"{config['dirs']['protonated_pdb']}/{pid}.pdb"

        crop_r = config['ppi_const']['crop_r']
        contact_d = config['ppi_const']['contact_d']
        out_file = config['dirs']['cropped_pdb'] + pid + '.pdb'

        if os.path.exists(out_file):
            # Skip if file already exists
            processed_ppi.append(ppi)
            continue

        res_dict_1 = get_coord_dict(pid, pdb_file, ch1)
        res_dict_2 = get_coord_dict(pid, pdb_file, ch2)

        if len(res_dict_1['res_id']) == 0 and len(res_dict_2['res_id']) == 0:
            # The PDB file is empty. Skpping...
            continue

        # Search for contact points within
        pdb_tree_1 = cKDTree(res_dict_1['atom_coord'])

        all_dist, all_idx_1 = pdb_tree_1.query(res_dict_2['atom_coord']) # all_idx_1 - index of the first array; len(all_indx_1) is the lenght if the second array

        contact_indx2 = np.where(all_dist<contact_d)
        contact_indx1 = np.unique(all_idx_1[contact_indx2])

        center_iface1 = np.mean(np.array(res_dict_1['atom_coord'])[contact_indx1], axis=0)
        center_iface2 = np.mean(np.array(res_dict_2['atom_coord'])[contact_indx2], axis=0)

        # Compute all residues to include
        dist_func1 = lambda x: (center_iface1[0] - x[0])**2 + (center_iface1[1] - x[1])**2 + (center_iface1[2] - x[2])**2
        dist_func2 = lambda x: (center_iface2[0] - x[0])**2 + (center_iface2[1] - x[1])**2 + (center_iface2[2] - x[2])**2

        dist_to_center1 = np.array([dist_func1(xi) for xi in res_dict_1['atom_coord']])
        dist_to_center2 = np.array([dist_func2(xi) for xi in res_dict_2['atom_coord']])

        min_dist1 = min(dist_to_center1) # if the center is outside of the surface, upper bound distance is crop_r + min_dist
        min_dist2 = min(dist_to_center2)

        res_to_include1_indx = np.where(dist_to_center1<(crop_r**2 + min_dist1**2))
        res_to_include2_indx = np.where(dist_to_center2<(crop_r**2 + min_dist2**2))

        res_to_include1 = [res_dict_1['chain_id'][i] + ':' + str(res_dict_1['res_id'][i]) for i in range(0, len(res_dict_1['chain_id']))]
        res_to_include1 = np.unique(np.array(res_to_include1)[res_to_include1_indx])

        res_to_include2 = [res_dict_2['chain_id'][i] + ':' + str(res_dict_2['res_id'][i]) for i in range(0, len(res_dict_2['chain_id']))]
        res_to_include2 = np.unique(np.array(res_to_include2)[res_to_include2_indx])

        res_to_include = np.append(res_to_include1, res_to_include2)

        with open(out_file, 'w') as out:
            with open(pdb_file, 'r') as f:
                for line in f.readlines():
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        ch = line[21]
                        res_id = int(line[22:26])

                        if '{}:{}'.format(ch, res_id) in res_to_include:
                            out.write(line)
                    else:
                        out.write(line)
        processed_ppi.append(ppi)

        extract_pdb_chain(out_file, config['dirs']['cropped_pdb'] + '/{}_{}.pdb'.format(pid, ch1), ch1)
        extract_pdb_chain(out_file, config['dirs']['cropped_pdb'] + '/{}_{}.pdb'.format(pid, ch2), ch2)

    return processed_ppi
