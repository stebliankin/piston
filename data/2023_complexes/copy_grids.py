import os
from tqdm import tqdm
data_prepare = './data_preparation/docked/'
complexes_2023 = [x.strip('\n') for x in open("./complexes2023.txt").readlines()]

out_grid = "./grid16R/"
if not os.path.exists(out_grid):
    os.mkdir(out_grid)

for ppi in tqdm(complexes_2023):
    grid_dir = data_prepare + ppi + '/07-grid_16R'
    os.system(f"cp {grid_dir}/* {out_grid}")
