import os

config = {}

config['dirs'] = {}
config['dirs']['data_prepare'] = os.getcwd()+ '/../../data/capri_score/piston_prepare_20R/'

config['dirs']['raw_pdb'] =  '/home/vsteb002/DeepDeltaAff/data/benchmark_deepRank/piston_prepare/' + '00-raw_pdbs/'
config['dirs']['protonated_pdb'] = '/home/vsteb002/DeepDeltaAff/data/benchmark_deepRank/piston_prepare/' + '01-protonated_pdb/'
config['dirs']['refined'] = config['dirs']['data_prepare'] + '02-refined_pdb/'
config['dirs']['cropped_pdb'] = config['dirs']['data_prepare'] + '03-cropped_pdbs/'
config['dirs']['chains_pdb'] = config['dirs']['data_prepare'] + '04-chains_pdbs/'

config['dirs']['surface_ply'] = config['dirs']['data_prepare'] + '05-surface_ply/'
#config['dirs']['patch_ply'] = config['dirs']['data_prepare'] + '05-patch_ply/'
config['dirs']['patches'] = config['dirs']['data_prepare'] + '06-patches/'
config['dirs']['grid'] = config['dirs']['data_prepare'] + '07-grid/'


config['dirs']['docked'] = config['dirs']['data_prepare'] + 'docked/'

#config['dirs']['neg_examples'] = os.getcwd()+'/../03-2022-prepare_docking/neg_examples/'
config['dirs']['dl_models'] = os.getcwd() + '/savedModels/'

config['dirs']['tmp'] = os.getcwd()+'/tmp/'

config['ppi_const'] = {}
config['ppi_const']['contact_d'] = 5 # minimum distance between residues to be considered as "contact point"
config['ppi_const']['surf_contact_r'] = 1 # minimum distance between two surface points to be considered as "contact point"
config['ppi_const']['patch_r'] = 16
config['ppi_const']['crop_r'] = config['ppi_const']['patch_r'] + 1 # radius to crop (in Angstroms)

config['ppi_const']['points_in_patch'] = 400

#config['ppi_const']['iRMSD_threshold'] = 5
# config['ppi_const']['iRMSD_threshold_pos'] = 5
# config['ppi_const']['iRMSD_threshold_neg'] = 10


config['mesh'] = {}
config['mesh']['mesh_res'] = 1.0 # resolution of the mesh

# Create Directories
for dir in config['dirs'].values():
    if not os.path.exists(dir):
        os.makedirs(dir)

os.environ["TMP"] = config['dirs']['tmp']
os.environ["TMPDIR"] = config['dirs']['tmp']
os.environ["TEMP"] = config['dirs']['tmp']
