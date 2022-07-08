import torch
import random
import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from FrEIA.modules import *
from FrEIA.framework import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from qPAI_cINN_uncertainty_estimation.data import MultiSpectralPressureO2Dataset
from qPAI_cINN_uncertainty_estimation.normalisation import (
    batch_spectrum_processing,
    spectrum_normalisation,
    spectrum_processing,
)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

if use_cuda:
    print("GPU available")
    data_path = Path("../datasets")
else:
    print("CPU only")
    data_path = Path(
        "C:\\Users\\dohert01\\PycharmProjects\\qPAI_cINN_uncertainty_estimation\\datasets"
    )

batch_size = 1
seq_length = 41
n_features = 1
lstm_hidden = 100
inn_hidden = 128
experiment_name = "FlowPhantom_insilico_complicated"

training_spectra_file = data_path / experiment_name / "training_spectra.pt"
validation_spectra_file = data_path / experiment_name / "validation_spectra.pt"
test_spectra_file = data_path / experiment_name / "test_spectra.pt"

training_oxygenations_file = data_path / experiment_name / "training_oxygenations.pt"
validation_oxygenations_file = (
    data_path / experiment_name / "validation_oxygenations.pt"
)
test_oxygenations_file = data_path / experiment_name / "test_oxygenations.pt"

train_spectra_original = torch.load(training_spectra_file)
train_oxygenations_original = torch.load(training_oxygenations_file)
validation_spectra_original = torch.load(validation_spectra_file)
validation_oxygenations_original = torch.load(validation_oxygenations_file)
test_spectra_original = torch.load(test_spectra_file)
test_oxygenations_original = torch.load(test_oxygenations_file)

# Zeroing out some of the spectrum data (randomly) and normalising
allowed_datapoints = [10]
train_spectra = batch_spectrum_processing(train_spectra_original, allowed_datapoints)
validation_spectra = batch_spectrum_processing(
    validation_spectra_original, allowed_datapoints
)
test_spectra = batch_spectrum_processing(test_spectra_original, allowed_datapoints)

# Reshaping initial pressure spectra to fit LSTM input size
train_spectra = torch.reshape(
    train_spectra, (len(train_spectra), len(train_spectra[0]), 1)
)
validation_spectra = torch.reshape(
    validation_spectra, (len(validation_spectra), len(validation_spectra[0]), 1)
)
test_spectra = torch.reshape(test_spectra, (len(test_spectra), len(test_spectra[0]), 1))

train_oxygenations = torch.reshape(
    train_oxygenations_original, (len(train_oxygenations_original), 1)
)
validation_oxygenations = torch.reshape(
    validation_oxygenations_original, (len(validation_oxygenations_original), 1)
)
test_oxygenations = torch.tensor(np.float32(test_oxygenations_original))
test_oxygenations = torch.reshape(
    test_oxygenations_original, (len(test_oxygenations_original), 1)
)

training_dataset = MultiSpectralPressureO2Dataset(
    train_spectra,
    train_oxygenations,
    transform=switch_seq_feat,
    target_transform=float_transform,
)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
