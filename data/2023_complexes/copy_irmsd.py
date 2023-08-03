## Copy qualities of docking models
import os
from tqdm import tqdm
import shutil

complexes_2023 = [x.strip('\n') for x in open("./complexes2023.txt").readlines()]
data_prepare = './data_preparation/docked/'

out_quality = "./docking_qualities/"
if not os.path.exists(out_quality):
    os.mkdir(out_quality)

for ppi in tqdm(complexes_2023):
    qual_path = f"{data_prepare}/{ppi}/irmsd.csv"
    out_qual = f"{out_quality}/{ppi}_qual.csv"
    shutil.copyfile(qual_path, out_qual)
