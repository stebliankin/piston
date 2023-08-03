import argparse
from data_prepare.data_prepare import prepare
#from utils.infer import infer_cmd
parser = argparse.ArgumentParser(prog="PIsToN")

parser.add_argument('--config', help='config file')


sp = parser.add_subparsers()

# Sub argparse [-prepare, -train, -eval]

sp_prepare = sp.add_parser('prepare', help='Data preparation module')
sp_prepare.add_argument('--list', help='List with PPIs in format PID_A_B')
sp_prepare.add_argument('--ppi', help='PPI in format PID_A_B (mutually exclusive with the --list option)')
sp_prepare.add_argument('--no_download',  default=False, action="store_true", help='If set True, the pipeline will skip the download part.')
sp_prepare.add_argument('--download_only',  default=False, action="store_true", help='If set True, the program will only download PDB structures without processing them.')
sp_prepare.add_argument('--prepare_docking',  default=False, action="store_true", help='If set True, re-dock ground truth structures and pre-process top 100 generated models')

sp_prepare.set_defaults(func=prepare)

args = parser.parse_args()
args.func(args)



