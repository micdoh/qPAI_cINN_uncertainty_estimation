import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import qPAI_cINN_uncertainty_estimation.config as c


def spectrum_normalisation(spectrum):
    """Applies z-score scaling to the initial pressure spectrum"""
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    norm = (spectrum - mean) / std
    return norm


def spectrum_processing(spectrum, allowed_datapoints):
    """Returns a normalised initial pressure spectrum with some of the values zeroed out"""
    num_non_zero_datapoints = random.choice(allowed_datapoints)
    a = np.zeros(len(spectrum))
    a[:num_non_zero_datapoints] = 1
    np.random.shuffle(a)

    incomplete_spectrum = list(np.multiply(a, np.array(spectrum)))
    non_zero_indices = np.nonzero(incomplete_spectrum)
    non_zero_values = list(filter(None, incomplete_spectrum))
    normalised_non_zero = spectrum_normalisation(non_zero_values)

    i = 0
    masking_features = [0] * len(
        spectrum
    )  # Binary to indicate if wavelength has been zeroed (or was zero already)
    for index in non_zero_indices[0]:
        incomplete_spectrum[index] = normalised_non_zero[i]
        masking_features[index] = 1
        i += 1

    normalised_incomplete_spectrum = np.array(incomplete_spectrum)
    masking_features = np.array(masking_features)

    return np.column_stack((normalised_incomplete_spectrum, masking_features))


def batch_spectrum_processing(batch, allowed_datapoints):
    processed = []

    for spectrum in batch:

        processed.append(spectrum_processing(spectrum, allowed_datapoints))

    return torch.tensor(np.array(processed))


def switch_seq_feat(tensor):
    # Return a view of the tensor with axes rearranged
    return torch.permute(tensor, (1, 0)).float()


def reshape_and_float(tensor):
    return torch.reshape(tensor, (c.seq_length, 2)).float()


def two_vector_float(tensor):
    return tensor.repeat(2).float()


class MultiSpectralPressureO2Dataset(Dataset):
    def __init__(self, spectra, oxygenations, transform=None, target_transform=None):
        self.data = spectra
        self.labels = oxygenations
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label


def prepare_dataloader(
    data_path: Path,
    experiment_name: str,
    train_val_test: str,
    allowed_datapoints: list,
    batch_size: int,
    transform=reshape_and_float,
    target_transform=two_vector_float,
):
    spectra_file = data_path / experiment_name / f"{train_val_test}_spectra.pt"
    oxygenations_file = (
        data_path / experiment_name / f"{train_val_test}_oxygenations.pt"
    )

    spectra_original = torch.load(spectra_file)
    oxygenations_original = torch.load(oxygenations_file)

    processed_spectra = batch_spectrum_processing(spectra_original, allowed_datapoints)
    processed_oxygenations = torch.reshape(
        oxygenations_original, (len(oxygenations_original), 1)
    )

    dataset = MultiSpectralPressureO2Dataset(
        processed_spectra,
        processed_oxygenations,
        transform=transform,
        target_transform=target_transform,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# Example calls of prepare_dataloader()
# training_dataloader = prepare_dataloader(c.data_path, c.experiment_name, 'training', c.allowed_datapoints, c.batch_size)
# validation_dataloader = prepare_dataloader(c.data_path, c.experiment_name, 'validation', c.allowed_datapoints, c.batch_size)
# test_dataloader = prepare_dataloader(c.data_path, c.experiment_name, 'test', c.allowed_datapoints, c.batch_size)
