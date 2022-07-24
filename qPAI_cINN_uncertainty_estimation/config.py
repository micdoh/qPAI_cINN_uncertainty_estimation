import torch
from pathlib import Path

"""
Global configuration for the experiments
Originally from: https://github.com/VLL-HD/analyzing_inverse_problems/
"""

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

if use_cuda:
    print("GPU available")
    data_path = Path("../datasets")
    output_dir = Path("../output")
    log_dir = Path("../logs")
else:
    print("CPU only")
    path = "C:\\Users\\dohert01\\PycharmProjects\\qPAI_cINN_uncertainty_estimation"
    data_path = Path(f"{path}\\datasets")
    output_dir = Path(f"{path}\\output")
    log_dir = Path(f"{path}\\logs")

use_default_model = False
batch_size = 1024
seq_length = 41
fcn_dim_out = 1
inn_input_dim = 2
lstm_input_dim = 2
lstm_hidden = 100
inn_hidden = 500
n_blocks = 8  # No. of invertible blocks in INN
total_data_dims = 2  # TODO - Should this be 41 or 82 (after masking)? Or 2!
use_fcn_layer = True
cond_length = seq_length * lstm_hidden if not use_fcn_layer else seq_length
experiment_name = "FlowPhantom_insilico_complicated"
allowed_datapoints = [40]#[10]
#load_date = "2022-07-12_11_08_52"  # Original, default model
#load_date = "2022-07-14_17_52_20"  # Larger model without FCN layer
load_date = "2022-07-21_09_39_35" # Wider model, less blocks, with FCN, 10 wavelengths train
clip_gradients = True
n_samples = 1000  # Number of samples for inference
sample_posterior = True  # Draw n_samples to find posterior, otherwise just sample at +/-1
visualisation = False if use_cuda else True
load_eval_data = False if use_cuda else True
save_eval_data = True if use_cuda else False
#load_eval_data_date = '2022-07-15_00_16_03'  # Larger model without FCN layer, 10 wavelengths eval
#load_eval_data_date = "2022-07-22_15_51_40"  # Wide model, less blocks, with FCN, 10 wavelengths eval
load_eval_data_date = '2022-07-24_18_14_43'  # Wide model, less blocks, with FCN, 40 wavelengths eval


#######################
#  Training schedule  #
#######################
eps = 1e-6
n_epochs = 1000
max_epochs = 2500
checkpoint_save_interval = 2501
no_improvement_epoch_cutoff = 80
adam_betas = (0.9, 0.95)
weight_decay = 1e-5
lr = 1e-3
decay_by = 0.01
gamma = decay_by ** (1.0 / n_epochs)

