import torch
import numpy as np
from pathlib import Path

"""
Global configuration for the experiments
Originally from: https://github.com/VLL-HD/analyzing_inverse_problems/
"""

#######################
# Path configuration  #
#######################
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

if use_cuda:
    print("GPU available")
    data_path = Path("../datasets")
    output_dir = Path("../output")
    log_dir = Path("../logs")
    torch.cuda.set_device(device)
else:
    print("CPU only")
    path = "C:\\Users\\dohert01\\PycharmProjects\\qPAI_cINN_uncertainty_estimation"
    data_path = Path(f"{path}\\datasets")
    output_dir = Path(f"{path}\\output")
    log_dir = Path(f"{path}\\logs")

    
#######################
#   Model parameters  #
#######################    
use_default_model = False
seq_length = 41
fcn_dim_out = 1
inn_input_dim = 2
lstm_input_dim = 2
lstm_hidden = 100
inn_hidden = 512
inn_subnet_layers = 1
n_blocks = 8  # No. of invertible blocks in INN
total_data_dims = 2  # Size of input/output data (not condition)
use_fcn_layer = False
cond_length = seq_length * lstm_hidden if not use_fcn_layer else seq_length*fcn_dim_out


#######################
#  Training schedule  #
#######################
batch_size = 2048
eps = 1e-6
min_epochs = 900
max_epochs = 1600
checkpoint_save_interval = max_epochs + 1
no_improvement_epoch_cutoff = 100
adam_betas = (0.9, 0.95)
weight_decay = 1e-5
lr = 1e-3
decay_by = 0.01
gamma = decay_by ** (1.0 / min_epochs)
clip_gradients = True


#######################
#    Other options    #
#######################
experiment_name = "NoSkin_filtered"
allowed_datapoints = [10]
partition_sparsity = True  # Partition spectra by sparsity with equal number of wavelengths in each
balance_dataset = False  # Use WeightedRandomSampler to ensure even distribution of sO2 labels in the dataset

n_samples = 1000  # Number of samples for inference

visualisation = False if use_cuda else True
load_eval_data = False if use_cuda else True
save_eval_data = True if use_cuda else False
load_for_retraining = False

#load_date = "2022-07-12_11_08_52"  # Original, default model, 10 wavelengths train
#load_date = "2022-07-14_17_52_20"  # Larger model without FCN layer, 10 wavelengths train
#load_date = "2022-07-21_09_39_35" # Wider model, less blocks, with FCN, 10 wavelengths train
#load_date = "2022-07-24_21_29_34"  # Wider model, less blocks, without FCN, 40 wavelengths train
load_date = '2022-07-25_16_13_12'  # Wider model, less blocks, without FCN, flexi-train

#load_eval_data_date = '2022-07-15_00_16_03'  # Larger model without FCN layer, 10 wavelengths eval
#load_eval_data_date = "2022-07-22_15_51_40"  # Wide model, less blocks, with FCN, 10 wavelengths eval
#load_eval_data_date = '2022-07-24_18_14_43'  # Wide model, less blocks, with FCN, 10 wavelengths train, 40 wavelengths eval
#load_eval_data_date = '2022-07-25_15_15_11'  # Wide model, less blocks, without FCN, 40 wavelengths train and eval
#load_eval_data_date = '2022-07-25_15_32_04'  # Wide model, less blocks, without FCN, 40 wavelengths train, 10 eval
#load_eval_data_date = '2022-07-26_14_45_48'  # Flexi, 10 eval
load_eval_data_date = '2022-07-27_16_58_04'  # Flexi, 40 eval
