import torch
import numpy as np
from networks.PIsToN_multiAttn import PIsToN_multiAttn
from networks.ViT_pytorch import get_ml_config
from utils.dataset import PISToN_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def infer(ppi_list, grid_dir, model_path, params, device, radius):
    """
    Obtain PIsToN scores
    ------------------------------------------------------------------------------
        ppi_list - list of protein complexes
        grid_dir - directory with pre-processed interface maps
        model_path - path to pre-train PIsToN model
        params - model parameters (dim_head, hidden_size, n_heads, transformer_depth)
        device - device to use for inference (ex. cpu)
        radius - radius of the patch (12A, 16A, or 20A)
    ------------------------------------------------------------------------------
    Return:
        output - score for each complex in ppi_list
        attn - list of attention maps for each complex in ppi_list
    """
    device = torch.device(device)

    model_config = get_ml_config(params)

    model = PIsToN_multiAttn(model_config, img_size=radius * 2,
                             num_classes=2).float().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Loaded PIsToN model with {n_params} trainable parameters. Radius of the patch: {radius}A")

    ## Constructing a dataset
    dataset = PISToN_dataset(grid_dir, ppi_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)

    all_outputs = []  # output score
    all_attn = []  # output attention map

    with torch.no_grad():
        for instance in tqdm(dataloader):
            grid, all_energies = instance
            grid = grid.to(device)
            all_energies = all_energies.float().to(device)
            model = model.to(device)
            output, attn = model(grid, all_energies)
            all_outputs.append(output)
            all_attn.append(attn)

    output = torch.cat(all_outputs, axis=0)
    return output, all_attn

def infer_cmd(args):
    """
    Obtain PIsToN scores (from command line)
    """
    if (not args.list and not args.ppi) or (args.list is not None and args.ppi is not None):
        raise AssertionError('Specify either "--list" or "--ppi" input')
    if (args.list is not None):
        ppi_list = [x.strip('\n') for x in open(args.list)]
    elif (args.ppi is not None):
        ppi_list = [args.ppi]

    print(f"Obtaining scores for {len(ppi_list)} complexes...")

    grid_dir = args.grid_dir
    model_path = args.model
    device = args.device
    params = args.model_params
    radius=args.radius

    infer(ppi_list, grid_dir, model_path, params, device, radius)








