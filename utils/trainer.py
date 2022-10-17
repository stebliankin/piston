"""
Objective:
   Utilities to train the PiSToN model.

Author:
    Vitalii Stebliankin (vsteb002@fiu.edu)
    Bioinformatics Research Group (BioRG)
    Florida International University

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from ray import tune

import json
import pdb

import time


#from utils.dataset import PDB_complex
import numpy as np

from tqdm import tqdm
from sklearn import metrics

#import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime

def get_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def set_device(model, device_ids, device):
    if device_ids is None and device is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
        model = model.to(device, non_blocking=False)
    elif device is not None:
        model = model.to(device, non_blocking=False)
    elif device_ids is not None:
        print("Setting up the following GPUs: {}".format(device_ids))
        device=torch.device("cuda:{}".format(device_ids[0]))
        model = nn.DataParallel(model, device_ids=device_ids).to(device, non_blocking=False)
        #model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    return model, device

def label_to_tensor(label, n_classes):
    # output shape (n_batch, n_classes]
    label_tensor = torch.zeros(label.shape[0], n_classes)
    for batch_i in range(label.shape[0]):
        for label_i in range(n_classes):
            label_tensor[batch_i][label_i]=label[batch_i]==label_i
    return label_tensor.long()

def add_to_history(history, train_loss, val_loss,train_auc, val_auc):
    history['train_loss'].append(float(train_loss))
    history['val_loss'].append(float(val_loss))
    history['train_auc'].append(float(train_auc))
    history['val_auc'].append(float(val_auc))

    return history

def compute_performance(output, label):
    # output - output weights from the model (unnormalized)
    # label - true label
    #pred_probabilities = F.softmax(output, dim=1)
    #pred_label = torch.argmax(pred_probabilities, dim=1)
    pred_probabilities = -output.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    #label_tensor = label_to_tensor(label, n_classes).detach().numpy()
    auc = metrics.roc_auc_score(label, pred_probabilities)
    return auc

def plot_metrics(history, saved_model_dir, model_name):
    plt.style.use('ggplot')

    figures_dir = saved_model_dir + '/' + model_name +'_figs/'
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)
    # plot losses:
    plt.plot(history['train_loss'], marker='.', color='b', label='Train loss')
    plt.plot(history['val_loss'], marker='.', color='r', label='Validation loss')
    plt.legend(loc="upper right")
    plt.savefig(figures_dir+'/loss_{}.png'.format(model_name))
    plt.clf()
    plt.plot(history['train_auc'], marker='.', color='b', label='Train AUC')
    plt.plot(history['val_auc'], marker='.', color='r', label='Validation AUC')
    plt.legend(loc="lower right")
    plt.savefig(figures_dir+'/auc_{}.png'.format(model_name))

def evaluate_val(loader, model, device, criterion=None, include_energy=False, include_attn=False, inside_loss=False):
    # Evaluating the model:
    running_loss, running_auc = 0, 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            #print("[{}] Loading image...".format(get_date()))
            if not include_energy:
                image_tiles, label, ppi = data
            else:
                image_tiles, energy_terms, label, ppi = data
                energy_terms = np.reshape(energy_terms,
                                          (energy_terms.shape[0] * energy_terms.shape[1], energy_terms.shape[2]))
                energy_terms = energy_terms.float().to(device, non_blocking=False)

            image_tiles = np.reshape(image_tiles, (image_tiles.shape[0]*image_tiles.shape[1], image_tiles.shape[2], image_tiles.shape[3], image_tiles.shape[4]))
            label = np.reshape(label, (label.shape[0]*label.shape[1]))
            #print("[{}] Transforming to device...".format(get_date()))
            image = image_tiles.to(device, non_blocking=False)
            label = label.to(device, non_blocking=False)
            #print("[{}] Running the model...".format(get_date()))
            if not include_energy:
                output = model(image)  # mask=background_mask
            elif inside_loss:
                output = model(image, energy_terms, label)
            else:
                output = model(image, energy_terms)
            if include_attn and inside_loss:
                output, attn, loss = output
            elif include_attn:
                output, attn = output
            elif inside_loss:
                output, loss = output
                # print("[{}] Computing loss...".format(get_date()))
                # pdb.set_trace()
            if not inside_loss:
                loss = criterion(output, label)
            #print("[{}] Computing loss...".format(get_date()))

            # loss = criterion(output, label)
            #print("[{}] Computing performance...".format(get_date()))
            auc = compute_performance(output, label)
           # print("[{}] Saving performance...".format(get_date()))

            running_loss += loss
            running_auc += auc


    val_loss = running_loss / len(loader)
    val_auc = running_auc / len(loader)

    return val_loss, val_auc


def train_one_epoch(model, train_loader, device, criterion, optimizer,
                    disable_tqdm=False, include_energy=False, include_attn=False, inside_loss=False):
    """

    :param model:
    :param train_loader:
    :param device:
    :param criterion:
    :param optimizer:
    :param disable_tqdm:
    :param include_energy:
    :param include_attn: if True, attention map is included in model output
    :param inside_loss: if True, the loss will be computed as part of the forward pass of the model
    :return:
    """
    # since = time.time()
    running_loss = 0
    running_auc = 0
    model.train()
  #  nan_ppis = []
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True, disable=disable_tqdm):
        # training phase:
        #print("[{}] Loading image...".format(get_date()))
        #image_tiles, label, ppi = data
        if not include_energy:
            image_tiles, label, ppi = data
        else:
            image_tiles, energy_terms, label, ppi = data
            energy_terms = np.reshape(energy_terms, (energy_terms.shape[0]*energy_terms.shape[1], energy_terms.shape[2]))
            energy_terms = energy_terms.float().to(device, non_blocking=False)


        image_tiles = np.reshape(image_tiles, (
        image_tiles.shape[0] * image_tiles.shape[1], image_tiles.shape[2], image_tiles.shape[3],
        image_tiles.shape[4]))
        label = np.reshape(label, (label.shape[0] * label.shape[1]))

        #print("[{}] Transforming to device...".format(get_date()))
        image = image_tiles.to(device=device, dtype=torch.float)
        label = label.to(device)
        #print("[{}] Running the model...".format(get_date()))
        if not include_energy:
            output = model(image)  # mask=background_mask
        elif inside_loss:
            output = model(image, energy_terms, label)
        else:
            output = model(image, energy_terms)
        if include_attn and inside_loss:
            output, attn, loss = output
        elif include_attn:
            output, attn = output
        elif inside_loss:
            output, loss = output
        #print("[{}] Computing loss...".format(get_date()))
        if not inside_loss:
            loss = criterion(output, label)
       # print("[{}] Computing performance...".format(get_date()))
       #  try:
        auc = compute_performance(output, label)
        # print("[{}] Backpropogation...".format(get_date()))
        loss.backward()
        optimizer.step()  # update weight
        optimizer.zero_grad()
        optimizer.step()

        # # print("[{}] Saving performance...".format(get_date()))
        running_loss += loss.item()

        running_auc += auc

    model.eval()
    train_loss = running_loss / len(train_loader)
    train_auc = running_auc / len(train_loader)
    print("Average training loss: {}; train AUC: {};".format(train_loss, train_auc))
    return model, train_loss, train_auc


def fit_supCon(epochs, model, train_loader, val_loader, optimizer, criterion=None, model_name='default', image_size=32, channels=10, device_ids = None,
        device=None, saved_model_dir='./savedModels/', save_model=True, print_summary = True, patience=10, raytune=False,
        disable_tqdm=False, include_energy=False, n_individual=None, include_attn=False, inside_loss=False):
    """

    :param epochs:
    :param model:
    :param train_loader:
    :param val_loader:
    :param optimizer:
    :param criterion: loss function
    :param model_name:
    :param image_size:
    :param channels:
    :param device_ids:
    :param device:
    :param saved_model_dir:
    :param save_model:
    :param print_summary:
    :param patience:
    :param raytune:
    :param disable_tqdm:
    :param include_energy: If True, your network should take as input both image and energy term
    :param include_attn: If True, attention map is included in the output
    :param inside_loss: if True, the loss will be computed as part of the forward pass of the model

    :return:
    """
    # WORKS ONLY WITH 2 CLASSES!!!
    start = time.time()
    if not os.path.exists(saved_model_dir):
        os.mkdir(saved_model_dir)
    elif os.path.exists(saved_model_dir + '/{}.pth'.format(model_name)):
        model.load_state_dict(torch.load(saved_model_dir + '/{}.pth'.format(model_name)))

    # device_ids - multiple devices
    # device - a single device name

    history = {'train_loss': [], 'val_loss': [],
               'train_auc':[], 'val_auc':[]
               }

    model, device = set_device(model, device_ids, device)

    # Plot model summary
    print("Start training {} model.".format(model_name))

    if print_summary:
        #summary(model, (channels, image_size, image_size))
        if not include_energy:
            summary(model, torch.rand((1, channels, image_size, image_size)).to(device, non_blocking=False))
        else:
            summary(model, torch.rand((1, channels, image_size, image_size)).to(device, non_blocking=False),
                    torch.rand((1, n_individual)).to(device, non_blocking=False))

    min_loss = np.inf
    max_auc = 0
    decrease = 0
    not_improved = 0
    saved_index = 0

    for e in range(epochs):
        print("[{}] Starting training for epoch {}...".format(get_date(), e))

        model, train_loss, train_auc = train_one_epoch(model, train_loader, device, criterion, optimizer,
                                                                  disable_tqdm=disable_tqdm, include_energy=include_energy,
                                                                  include_attn=include_attn, inside_loss=inside_loss)

        print("Evaluating the model on validation set...".format(e))
        val_loss, val_auc = evaluate_val(val_loader, model, device, criterion,
                                                                                  include_energy=include_energy,
                                                                                  include_attn=include_attn, inside_loss=inside_loss)
        print("Average val loss: {}; val AUC: {};".format(val_loss, val_auc))

        #if val_loss<min_loss:
        if val_auc>max_auc:
            #print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, val_loss))
            print('AUC increasing.. {:.4f} >> {:.4f} '.format(max_auc, val_auc))
            #min_loss = val_loss
            max_auc = val_auc
            decrease += 1
            print('saving model on epoch {}...'.format(e))
            torch.save(model.state_dict(), saved_model_dir + '/{}.pth'.format(model_name))
            saved_index = e
            not_improved=0
            if raytune:
                #tune.report(score=float(val_loss))
                tune.report(score=float(val_auc))

        else:
            not_improved += 1
            print("Model did not improved {} times...".format(not_improved))

        history = add_to_history(history, train_loss, val_loss, train_auc, val_auc)
        # pred_label = F.softmax(output, dim=1) loss function should take unnormalized scores as input

        if not_improved==patience:
            print("Stopping training...")
            break

    # Load lastly saved model
    model.load_state_dict(torch.load(saved_model_dir + '/{}.pth'.format(model_name)))

    print("[{}] Done with training.".format(get_date()))
    print("The model was saved at the {} epoch.".format(saved_index))
    print("Total training time: {} seconds.".format(time.time() - start))
    plot_metrics(history, saved_model_dir, model_name)
    with open(saved_model_dir+'/history.json', 'w') as outfile:
        json.dump(history, outfile)

    return model, history, saved_index


def get_processed(alist, config):
    pos_dir = config['dirs']['grid']
    processed_list = []
    for ppi in alist:
        if os.path.exists(pos_dir+ppi+'.npy'):
            processed_list.append(ppi)
    return processed_list
