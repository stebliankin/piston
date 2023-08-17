from datetime import datetime
from importlib.machinery import SourceFileLoader
import os
import pdb
from subprocess import Popen, PIPE
import shutil
import numpy as np

def get_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def read_config(args):
    # Read config
    if not args.config:
        from config_default import config
    else:
        config_module = SourceFileLoader("config", args.config).load_module()
        config = config_module.config

    print("[ {} ] Configuration parameters:".format(get_date()))
    print(config)
    return config

def get_processed(ppi_list, config):
    processed_ppis = []
    for ppi in ppi_list:
        pid, ch1, ch2 = ppi.split('_')
        if os.path.exists(config['dirs']['grid'] + '/' + ppi + '.npy'):
            processed_ppis.append(ppi)
    return processed_ppis
#

def learn_background_mask(grid):
    """
    Returns the mask with zero elements outside the patch
    :param grid: example of a grid image
    :return: mask
    """
    mask = np.zeros((grid.shape[0], grid.shape[1]))
    radius = grid.shape[0]/2
    for row_i in range(grid.shape[0]):
        for column_i in range(grid.shape[1]):
            # Check if coordinates are within the radius
            x = column_i - radius
            y = radius - row_i
            if x ** 2 + y ** 2 <= radius ** 2:
                mask[row_i][column_i] = 1
    return mask

def read_energies(energies_path, assign_zeros=False):
    # energies_path - path to the energy file computed by firedock
    # assign_zeros - if set True and energies file is empty, assign zeros to all energies (default is False)

    """
    :param ppi:
    :return: numpy array of energy terms:
        (0) - indx
        (1) - Lrmsd     - ligand rmsd of the final position, after the rigid-body optimization.
        (2) -Irmsd     - interface rmsd of the final position, after the rigid-body optimization.
        (3) - st_Lrmsd  - initial ligand rmsd.
        (4) - st_Irmsd  - initial ligand rmsd.
    0 - (5) - glob      - global score of the candidate, which is linear combination of the terms described bellow. To rank the candidates, you should sort the rows by this column in ascending order.
    1 - (6) - aVdW      - attractive van der Waals
    2 - (7) - rVdW      - repulsive van der Waals
    3 - (8) - ACE       - Atomic Contact Energy
    4 - (9) - inside    - "Insideness" measure, which reflects the concavity of the interface.
    5 - (10) - aElec     - short-range attractive electrostatic term
    6 - (11) - rElec     - short-range repulsive electrostatic term
    7 - (12) - laElec    - long-range attractive electrostatic term
    8 - (13) - lrElec    - long-range repulsive electrostatic term
    9 - (14) - hb        - hydrogen and disulfide bonding
    10 - (15) - piS	  - pi-stacking interactions
    11 - (16) - catpiS	  - cation-pi interactions
    12 - (17) - aliph	  - aliphatic interactions
         (18) - prob      - rotamer probability

    """
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
    if all_energies is not None:
        all_energies = np.nan_to_num(all_energies)
    elif assign_zeros:
        all_energies = np.zeros(13)

    return all_energies

def fix_residue_numbers(ppi, config):
    pid, ch1, ch2 = ppi.split('_')
    pdb_file = config['dirs']['protonated_pdb']+"/"+pid+".pdb"
    pdb_tmp_file = config['dirs']['protonated_pdb']+"/"+pid+"_tmp.pdb"
    shutil.copyfile(pdb_file, pdb_tmp_file)
    prev_resid=''
    prev_resname=''
    rename_flag=False
    all_latters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    letter_i = 0
    with open(pdb_file, 'w') as out:
        with open(pdb_tmp_file, 'r') as f:
            for line in f.readlines():
                if line[:4]=='ATOM':
                    curr_resname = line[17:20]
                    curr_resid = line[22:26]
                    if curr_resid==prev_resid and curr_resname!=prev_resname:
                        rename_flag=True
                        letter_i+=1
                    if curr_resid!=prev_resid:
                        rename_flag=False
                        letter_i=-1
                    if rename_flag:
                        line_list = [ch for ch in line]
                        line_list[26] = all_latters[letter_i]
                        # shift right the rest
                        for i in range(len(curr_resid)):
                            line_list[25-i] = curr_resid[-i-1]
                        line = ''.join(line_list)
                    prev_resid = curr_resid
                    prev_resname = curr_resname

                out.write(line)


    return

def merge_chains(pdb_in, ch1, ch2, pdb_out):
    # Chains from the first protein will be renamed to A, while second protein will be renamed to Z

    with open(pdb_in, 'r') as f:
        with open(pdb_out, 'w') as out:
            for line in f.readlines():
                if line[:6] == 'HEADER':
                    continue
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    line = [char for char in line]
                    if line[21] in ch1:
                        line[21] = 'Z'
                    elif line[21] in ch2:
                        line[21] = 'A'
                    line = ''.join(line)
                out.write(line)
    return None

def rename_chains(pid, ch, chains_pdb_dir, reversed=True):
    # Rename the chains to avoid names overlap during the docking step.
    # pid (str) - PID ID
    # ch (str) - current chain names
    # chains_pdb_dir (str) - directory with PDB files
    # reversed (bool) - if true, chains will be renamed in the descending lexicographic order (i.e. ZYXW...)

    # To avoid chains collision, the target chains will be renamed to "ABC"
    # Ligand chains will be renamed to "ZYX"

    # Write to the current directory
    chains_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    all_chains = chains_choices[:]
    if reversed:
        all_chains.reverse()

    PDB_TARGET = '{}_{}.pdb'.format(pid, ch)
    pdb_target_path = chains_pdb_dir + PDB_TARGET
    new_chains = []
    chains_seen = []
    with open('./' + PDB_TARGET, 'w') as out:
        with open(pdb_target_path, 'r') as f:
            for line in f.readlines():
                if line[:6] == 'HEADER':
                    continue
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    line = [char for char in line]
                    if line[21] not in chains_seen:
                        new_chains.append(all_chains.pop())
                        chains_seen.append(line[21])
                    line[21] = new_chains[-1]
                    line = ''.join(line)
                out.write(line)
    return './' + PDB_TARGET, ''.join(new_chains)

def extract_model(pdb_file, out_pdb, i):
    to_write=False
    # renamed_ch = None

    # prev_chain = None
    with open(pdb_file, 'r') as f:
        with open(out_pdb, 'w') as out:
            for line in f.readlines():
                # if line[:6] == 'HEADER':
                #     continue
                if line[:6]=='ENDMDL':
                    to_write=False
                if to_write: # if the right model
                    out.write(line)
                if line[:5]=='MODEL':
                    if line.split(' ')[-1].strip('\n') == str(i):
                        to_write = True
                    else:
                        to_write = False

def reset_config(config, new_dir):
    config['dirs']['data_prepare'] = new_dir

    for dir_key in config['dirs'].keys():
        if dir_key not in ['data_prepare', 'savedModels']:
            old_dir = config['dirs'][dir_key]
            base_dir = old_dir.split('/')[-1] if old_dir[-1]!='/' else old_dir.split('/')[-2]
            config['dirs'][dir_key] = new_dir + '/' + base_dir + '/'

    for dir in config['dirs'].values():
        if not os.path.exists(dir):
            os.makedirs(dir)
    return config

def fill_opacity(ppi, config):
    # fill the opacity of pdb files to one.
    # HDOCK do not produce opacity scores as output which causes errors in the MaSIF data preparee module
    pid, ch1, ch2 = ppi.split('_')

    with open(config['dirs']['protonated_pdb']+pid+'.pdb', 'w') as out:
        with open(config['dirs']['protonated_pdb']+pid+'_tmp.pdb', 'r') as f:
            for line in f.readlines():
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    #pdb.set_trace()
                    line = line[:55] + ' 1.00' + line[60:] + '\n'
                    # line[57:60] = '1.00'
                    # line = ''.join(line)
                out.write(line)
    os.remove(config['dirs']['protonated_pdb']+pid+'_tmp.pdb')

def extract_pdb_chain(pdb_full_file, pdb_chain_file, ch):
    with open(pdb_full_file, 'r') as f:
        with open(pdb_chain_file, 'w') as out:
            for line in f.readlines():
                if (line[0:4]=='ATOM' or line[0:6]=='HETATM') and line[21] in ch:
                    out.write(line)

def run_firedock_one(ppi, ppi_dir, firedock_out_dir):
    firedock_out_dir = firedock_out_dir + '/' + ppi + '/'
    pid, ch1, ch2 = ppi.split('_')
    ## Separate chains
    ch1_file =  f"{ppi_dir}/{pid}_{ch1}.pdb"
    ch2_file = f"{ppi_dir}/{pid}_{ch2}.pdb"
    extract_pdb_chain(f"{ppi_dir}/{pid}.pdb", ch1_file, ch1)
    extract_pdb_chain(f"{ppi_dir}/{pid}.pdb", ch2_file, ch2)

    # if not os.path.exists(f"{firedock_out_dir}/refined-out-{ppi}.ref"):
    ## Fix PDB files
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    firedock_params_dir = f"/FireDock/"

    if not os.path.exists(firedock_out_dir):
        os.mkdir(firedock_out_dir)

    args = [f'{firedock_params_dir}/preparePDBs.pl', ch1_file, ch2_file]
    print(' '.join(args))
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)

    ## Generate config for FireDock
    # /FireDock/buildFireDockParams.pl 1GL0_E.pdb.CHB.pdb 1GL0_I.pdb.CHB.pdb U U Default 1GLO.trans 1GL0.firedock.out 0 50 0.85 1 FireDock_params.txt
    args = ["/FireDock/buildFireDockParams.pl", f"{ch1_file}.CHB.pdb", f"{ch2_file}.CHB.pdb", "U", "U", "Default",
            f"{firedock_params_dir}/default.trans", f"{firedock_out_dir}/refined-out-{ppi}", '0', '50', '0.8', '1',
                                                    f"{firedock_out_dir}/params-{ppi}.txt"]
    print(' '.join(args))
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)

    ## Run FireDock
    # singularity exec ${DeepDeltaAff_PATH}/env/TransBind.sif /FireDock/runFireDock.pl FireDock_params.txt
    args = ["/FireDock/runFireDock.pl", f"{firedock_out_dir}/params-{ppi}.txt"]
    print(' '.join(args))
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)

    return 1

def refine_one(ppi, config):
    # $DeepDeltaAff_PATH/src/firedock_params/preparePDBs.pl 1GL0_E.pdb 1GL0_I.pdb
    run_firedock_one(ppi, config['dirs']['protonated_pdb'], f"{config['dirs']['refined']}")
    # Copy refined energies to the
    if os.path.exists(f"{config['dirs']['refined']}/{ppi}/refined-out-{ppi}.ref"):
        shutil.copyfile(f"{config['dirs']['refined']}//{ppi}/refined-out-{ppi}.ref",
                        f"{config['dirs']['grid']}/refined-out-{ppi}.ref")


def refine(ppis, config):
    """
    Run FireDock refinement and energy calculation.
    Rereference:
    Andrusier, Nelly, Ruth Nussinov, and Haim J. Wolfson.
                            "FireDock: fast interaction refinement in molecular docking."
                                Proteins: Structure, Function, and Bioinformatics 69.1 (2007): 139-159.
    :param ppis: list of PPIs
    :param config: dictionary with configurations
    :return: processed_ppis - list of processed PPIs
    """
    processed_ppi = []
    for ppi in ppis:
        # $DeepDeltaAff_PATH/src/firedock_params/preparePDBs.pl 1GL0_E.pdb 1GL0_I.pdb
        try:
            run_firedock_one(ppi, config['dirs']['protonated_pdb'], f"{config['dirs']['refined']}")
            # Copy refined energies to the
            processed_ppi.append(ppi)
            if os.path.exists(f"{config['dirs']['refined']}/{ppi}/refined-out-{ppi}.ref"):
                shutil.copyfile(f"{config['dirs']['refined']}//{ppi}/refined-out-{ppi}.ref",
                                f"{config['dirs']['grid']}/refined-out-{ppi}.ref")
        except Exception as e:
            print(e)
    return processed_ppi

def execute_hdock(pid1, ch1, pid2, ch2, new_ch1, new_ch2,  PDB_TARGET, PDB_LIGAND):
    out_pid = '{}-{}-{}-{}_{}_{}'.format(pid1, ch1, pid2, ch2, new_ch1, new_ch2)
    # pdb.set_trace()
    if not os.path.exists('{}.out'.format(out_pid)):
        args = ['hdock', PDB_TARGET, PDB_LIGAND, '-out', '{}.out'.format(out_pid)]
        print(' '.join(args))
        process = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)
    else:
        print('{}.out already exists.'.format(out_pid))
    return out_pid

def run_hdock_one(pid1, ch1, pid2, ch2, out_dock, config):
    if not os.path.exists(out_dock):
        os.mkdir(out_dock)

    print("[ {} ] Start docking {} with {}...".format(get_date(), pid1, pid2))

    # change dir to docking path
    curr_dir = os.getcwd()
    os.chdir(out_dock)

    # renaming chains
    PDB_TARGET, new_ch1 = rename_chains(pid1, ch1, config['dirs']['chains_pdb'])
    PDB_LIGAND, new_ch2 = rename_chains(pid2, ch2, config['dirs']['chains_pdb'], reversed=False)

    out_pid = execute_hdock(pid1, ch1, pid2, ch2, new_ch1, new_ch2,  PDB_TARGET, PDB_LIGAND)

    if not os.path.exists('{}.out'.format(out_pid)):
        # If docking failed, run the cropped version
        PDB_TARGET, new_ch1 = rename_chains(pid1, ch1, config['dirs']['cropped_pdb'])
        PDB_LIGAND, new_ch2 = rename_chains(pid2, ch2, config['dirs']['cropped_pdb'], reversed=False)
        out_pid = execute_hdock(pid1, ch1, pid2, ch2, new_ch1, new_ch2, PDB_TARGET, PDB_LIGAND)

    if not os.path.exists('{}.pdb'.format(out_pid)):
        args_pr = ['createpl', '{}.out'.format(out_pid), '{}.pdb'.format(out_pid), '-complex', '-nmax', '100']
        process = Popen(args_pr, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
    return out_pid, new_ch1, new_ch2
#['createpl 2NBV-A-2NBV-B_A_Z.out 2NBV-A-2NBV-B_A_Z.pdb -complex -nmax 100'

def combine_pdb(pdb1, pdb2, out_pdb, pdb_dir):
    # Combine two PDB files into one
    with open(pdb_dir+out_pdb, 'w') as out:
        for pdb_file in [pdb1, pdb2]:
            with open(pdb_dir+pdb_file, 'r') as f1:
                for line in f1.readlines():
                    line = line.strip('\n').strip('new').strip(' ')
                    out.write(line+'\n')