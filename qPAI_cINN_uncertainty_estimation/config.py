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

batch_size = 2048
seq_length = 41
n_features = 1
inn_input_dim = 2
lstm_hidden = 100
inn_hidden = 128
n_blocks = 12  # No. of invertible blocks in INN
total_data_dims = 2  # TODO - Should this be 41 or 82 (after masking)? Or 2!
experiment_name = "FlowPhantom_insilico_complicated"
allowed_datapoints = [10]
load_date = "2022-07-12_11_08_52"
clip_gradients = True
#######################
#  Training schedule  #
#######################
eps = 1e-6
n_epochs = 101
checkpoint_save_interval = 20
adam_betas = (0.9, 0.95)
weight_decay = 1e-5
lr = 1e-3
decay_by = 0.01
gamma = decay_by ** (1.0 / n_epochs)

