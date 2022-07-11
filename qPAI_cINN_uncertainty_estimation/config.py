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
    output_file = Path("../output/cinn")
    log_dir = Path("../logs")
else:
    print("CPU only")
    path = "C:\\Users\\dohert01\\PycharmProjects\\qPAI_cINN_uncertainty_estimation"
    data_path = Path(f"{path}\\datasets")
    output_dir = Path(f"{path}\\output")
    output_file = Path(f"{path}\\output\\cinn")
    log_dir = Path(f"{path}\\logs")

batch_size = 2048
seq_length = 41
n_features = 1
inn_input_dim = 2
lstm_hidden = 100
inn_hidden = 128
n_blocks = 12  # No. of invertible blocks in INN
total_data_dims = 41
experiment_name = "FlowPhantom_insilico_complicated"
allowed_datapoints = [10]
loss_display_cutoff = 10  # cut off the loss so the plot isn't ruined

######################
#  General settings  #
######################

# Filename to save the model under
# output_file = "output/my_inn.pt"
# Model to load and continue training. Ignored if empty string
filename_in = ""
# Use interactive visualization of losses and other plots. Requires visdom
interactive_visualization = True
# Run a list of python functions at test time after eacch epoch
# See toy_modes_train.py for reference example
test_time_functions = []

#######################
#  Training schedule  #
#######################
eps = 1e-6
# Initial learning rate
# lr_init = 1.0e-3
# Batch size
# batch_size = 500
# Total number of epochs to train for
n_epochs = 100
checkpoint_save_interval = 20
# End the epoch after this many iterations (or when the train loader is exhausted)
# n_its_per_epoch = 200
# For the first n epochs, train with a much lower learning rate. This can be
# helpful if the model immediately explodes.
# pre_low_lr = 0
# Decay exponentially each epoch, to final_decay*lr_init at the last epoch.
# final_decay = 0.02
# L2 weight regularization of model parameters
# l2_weight_reg = 1e-5
# Parameters beta1, beta2 of the Adam optimizer
adam_betas = (0.9, 0.95)

weight_decay = 1e-5
lr = 1e-3
decay_by = 0.01
gamma = decay_by ** (1.0 / n_epochs)

#####################
#  Data dimensions  #
#####################


############
#  Losses  #
############


###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.10
#
N_blocks = 4
#
exponent_clamping = 4.0
#
hidden_layer_sizes = 128
#
use_permutation = True
#
verbose_construction = False
