from .get_structure import protonate_pdb
from .get_structure import crop_pdb_one, download
from .triangulate import triangulate_single
from .compute_patches import compute_patches
from .convert_to_images import convert_to_images
from .map_patch_atom import map_patch_atom

from importlib.machinery import SourceFileLoader
from datetime import datetime
import time
from utils.utils import read_config, get_date, run_hdock_one, extract_model, reset_config, fill_opacity
from utils.utils import combine_pdb, merge_chains, refine_one, fix_residue_numbers

import pdb
import os
import shutil
from pdb2sql import StructureSimilarity


def triangulate_unrefined(ppi, config):
    print(f"WARNING::Couldn't trinagulate {ppi}.")
    print("Re-attempting with the unfixed file...")
    crop_pdb_one(ppi, config, use_refined=False)
    triangulate_single(ppi, config, overwrite=True)

def preprocess(processed_ppi, config):
    print("[ {} ] Downloaded {} complexes.".format(get_date(), len(processed_ppi)))

    for ppi in processed_ppi:

        out_grid = config['dirs']['grid'] + '/' + ppi + '.npy'
        if os.path.exists(out_grid):
            print(f"{out_grid} already exists...")
            continue

        refine_one(ppi, config)
        if 'refine' in config.keys() and not config['refine']:
            print("Refined complex will be used only for energy computation.")
            crop_pdb_one(ppi, config, use_refined=False)
        else:
            crop_pdb_one(ppi, config)

        try:
            triangulate_single(ppi, config)
        except:
            triangulate_unrefined(ppi, config)

        try:
            compute_patches([ppi], config)
        except:
            triangulate_unrefined(config)
            compute_patches([ppi], config)

        map_patch_atom([ppi], config)

        #convert_to_images([ppi], config)
        try:
            convert_to_images([ppi], config)
        except:
            print(f"WARNING::Couldn't convert to image for {ppi}.")
            print("Attempting to fix the file...")
            #download([ppi], config)
            #fix_residue_numbers(ppi, config)
            crop_pdb_one(ppi, config, use_refined=False)
            triangulate_single(ppi, config, overwrite=True)
            compute_patches([ppi], config, overwrite=True)
            map_patch_atom([ppi], config)

            convert_to_images([ppi], config)



def check_processed(ppis, config):
    processed = []
    for ppi in ppis:
        if os.path.exists(config['dirs']['grid']+'{}.npy'.format(ppi)):
            processed.append(ppi)
    return processed

def prepare_docking(ppis, config):
    # run docking on each ppi;
    # generate 100 complexes
    for ppi in ppis:

        pid, ch1, ch2 = ppi.split('_')
        out_dock = config['dirs']['docked'] + ppi
        out_pid, new_ch1, new_ch2 = run_hdock_one(pid, ch1, pid, ch2, out_dock, config)
        os.chdir(out_dock)

        if os.path.exists('irmsd.csv') and len(open('irmsd.csv', 'r').readlines())==101:
            continue

        config = reset_config(config, out_dock+'/')

        combine_pdb(pid + '_' + ch1 + ".pdb", pid + '_' + ch2 + ".pdb", "ref.pdb", './')

        # if os.path.exists(config['dirs']['grid']):
        #     shutil.rmtree(config['dirs']['grid'])
        #     os.mkdir(config['dirs']['grid'])

        with open('irmsd.csv', 'w') as out:
            out.write('model_i,model_PPI,iRMSD,lRMSD,FNAT\n')
            for model_i in range(1, 101):  # generate each model

                try:
                    model_ppi = '{}-model-{}_{}_{}'.format(pid, model_i, new_ch1, new_ch2)

                    extract_model('{}.pdb'.format(out_pid),
                                  config['dirs']['protonated_pdb'] + '{}-model-{}_tmp.pdb'.format(pid, model_i), model_i)
                    fill_opacity(model_ppi, config)

                    # Pre-compute features for each model
                    if not os.path.exists(config['dirs']['grid']+model_ppi+'.npy'):
                        preprocess([model_ppi], config)

                    # Temporary create dimers (i.e. merge chains for a single protein, if more than one).
                    # Merging is necessary because FNAT calculation requires dimers.
                    curr_pdb_docked = config['dirs']['protonated_pdb'] + '/{}-model-{}.pdb'.format(pid, model_i)
                    curr_pdb_ref = config['dirs']['data_prepare']+'ref.pdb'
                    tmp_pdb_docked = config['dirs']['tmp'] + '/{}-model-{}.pdb'.format(pid, model_i)
                    tmp_pdb_ref = config['dirs']['tmp'] + '/ref_{}'.format(pid)

                    merge_chains(curr_pdb_docked, new_ch2, new_ch1, tmp_pdb_docked)
                    merge_chains(curr_pdb_ref, new_ch2, new_ch1, tmp_pdb_ref)

                    sim = StructureSimilarity(tmp_pdb_docked, tmp_pdb_ref)

                    irmsd = sim.compute_irmsd_pdb2sql(method='svd')
                    lrmsd = sim.compute_lrmsd_pdb2sql(method='svd')
                    fnat = sim.compute_fnat_pdb2sql()

                    os.remove(tmp_pdb_docked)
                    os.remove(tmp_pdb_ref)


                    # if irmsd<config['ppi_const']['iRMSD_threshold']:
                    #     shutil.move(config['dirs']['grid']+model_ppi+'.npy', pos_dir+model_ppi+'.npy')
                    # else:
                    #     shutil.move(config['dirs']['grid'] + model_ppi + '.npy', neg_dir + model_ppi + '.npy')
                    out.write('{},{},{},{},{}\n'.format(model_i, model_ppi, irmsd, lrmsd, fnat))
                except:
                    pass

 #       shutil.rmtree(config['dirs']['protonated_pdb'])
        shutil.rmtree(config['dirs']['cropped_pdb'])
        shutil.rmtree(config['dirs']['chains_pdb'])
        shutil.rmtree(config['dirs']['surface_ply'])
        shutil.rmtree(config['dirs']['patches'])
        shutil.rmtree(config['dirs']['refined'])

def fix_protonated(ppi_list, config):
    for ppi in ppi_list:
        pid, ch1, ch2 = ppi.split('_')
        pdb_file = config['dirs']['protonated_pdb'] + pid + '.pdb'
        pdb_tmp_file = config['dirs']['protonated_pdb'] + "/" + pid + "_tmp.pdb"
        shutil.copyfile(pdb_file, pdb_tmp_file)
        with open(config['dirs']['protonated_pdb'] + pid + '.pdb', 'w') as out:
            with open(config['dirs']['protonated_pdb'] + pid + '_tmp.pdb', 'r') as f:
                for line in f.readlines():
                    if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                        if line[72]!=' ':
                            line = line[:21] + line[72] + line[22:72] + ' \n'

                    out.write(line)

        os.remove(config['dirs']['protonated_pdb'] + pid + '_tmp.pdb')

def prepare(args):
    """
    Data prepare module
    """
    start = time.time()

    print("[ {} ] Start data prepare...".format(get_date()))

    ppi_list = []

    if (not args.list and not args.ppi) or (args.list is not None and args.ppi is not None):
        raise AssertionError('Specify either "--list" or "--ppi" input')

    if (args.list is not None):
        ppi_list = [x.strip('\n') for x in open(args.list)]
    elif (args.ppi is not None):
        ppi_list = [args.ppi]

    # Read config
    config = read_config(args)

    print("[ {} ] Configuration parameters:".format(get_date()))
    print(config)

    #######################################################################################################################
    # Protonate PDB files
    #######################################################################################################################
    #print("[ {} ] Protonated PDB directory is set to {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), config['dirs']['raw_pdb_dir']))


    print("[ {} ] Preprocessing {} complexes".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), len(ppi_list)))

    if not args.no_download:
        processed_ppi = download(ppi_list, config)
    else:
        processed_ppi = ppi_list

    if args.fix_pdb:
        fix_protonated(ppi_list, config)

    if not args.download_only:
        preprocess(processed_ppi, config)

    if args.prepare_docking:
        prepare_docking(processed_ppi, config)

    #
    # ###############################################################################################################
    # # Debugging
    # ###############################################################################################################
    # from tmp.save_patch_mesh import save_patch_mesh
    #
    # patch_dir = config['dirs']['patches'] + ppi.split('_')[0] + '/'
    # ply_dir = config['dirs']['surface_ply']
    # save_patch_mesh(ppi, 1, 3138, ply_dir, patch_dir)
    # save_patch_mesh(ppi, 2, 655, ply_dir, patch_dir)
    # exit()

    print("[ {} ] The data preparation is complete.".format(get_date()))


    print("Total execution time for data preparation: {:.2f}m".format((time.time() - start)/60))