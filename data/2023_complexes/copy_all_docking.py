import shutil
import os
from tqdm import tqdm

ppi_list = [x.strip('\n') for x in open('list_complexes.txt', 'r').readlines()]
out_dir = './docking_models_2023/'
os.makedirs(out_dir, exist_ok=True)

remove_list = []

for ppi in ppi_list:
    doc_dir = f'./data_preparation/docked/{ppi}/01-protonated_pdb/'
    pid, ch1, ch2 = ppi.split('_')

    for i in range(100):
        if not os.path.exists(f'{doc_dir}/{pid}-model-{i+1}.pdb') and ppi not in remove_list:
            remove_list.append(ppi)
        #shutil.copyfile(f'{doc_dir}/{pid}-model-{i+1}.pdb', f'{out_dir}/{pid}-model-{i+1}.pdb')


print(remove_list)
ppi_list = [x for x in ppi_list if x not in remove_list]
print(len(ppi_list))

# with open('complexes2023.txt', 'w') as out:
#     for ppi in tqdm(ppi_list):
#         doc_dir = f'./data_preparation/docked/{ppi}/01-protonated_pdb/'
#         pid, ch1, ch2 = ppi.split('_')
#
#         for i in range(100):
#             shutil.copyfile(f'{doc_dir}/{pid}-model-{i+1}.pdb', f'{out_dir}/{pid}-model-{i+1}.pdb')
#         out.write(ppi+'\n')