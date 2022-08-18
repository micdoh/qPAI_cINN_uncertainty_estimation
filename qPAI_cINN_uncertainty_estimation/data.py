import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import qPAI_cINN_uncertainty_estimation.config as c


def load_spectra_file(file_path: str) -> tuple:
    print("Loading data...")
    data = np.load(file_path, allow_pickle=True)
    wavelengths = data["wavelengths"]
    oxygenations = data["oxygenations"]
    spectra = data["spectra"]
    distances = data["distances"]
    depths = data["depths"]
    melanin_concentration = data["melanin_concentration"]
    background_oxygenation = data["background_oxygenation"]
    if "timesteps" in data:
        timesteps = data["timesteps"]
    else:
        timesteps = None
    if "tumour_mask" in data:
        tumour_mask = data["tumour_mask"]
    else:
        tumour_mask = None
    if "reference_mask" in data:
        reference_mask = data["reference_mask"]
    else:
        reference_mask = None
    if "mouse_body_mask" in data:
        mouse_body_mask = data["mouse_body_mask"]
    else:
        mouse_body_mask = None
    if "background_mask" in data:
        background_mask = data["background_mask"]
    else:
        background_mask = None
    if "lu" in data:
        lu = data["lu"]
    else:
        lu = None

    print("Loading data...[DONE]")
    return (wavelengths, oxygenations, lu, spectra, melanin_concentration,
            background_oxygenation, distances, depths, timesteps,
            tumour_mask, reference_mask,
            mouse_body_mask, background_mask)


def spectrum_normalisation(spectrum):
    """Applies z-score scaling to the initial pressure spectrum"""
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    norm = (spectrum - mean) / std
    return norm


def set_ones(zeros, n_ones):
    zeros[:n_ones] = 1
    return zeros


def spectrum_processing(spectrum, allowed_datapoints):
    """Returns a normalised initial pressure spectrum with some of the values zeroed out
    N.B. the spectrum can be partitioned by the number of non-zero datapoints,
    so that each partition contains an equal (or almost equal) amount of non-zero values.
    The non-zero values within each partition are randomly distributed."""
    num_non_zero_datapoints = random.choice(allowed_datapoints)
    if num_non_zero_datapoints > len(spectrum):
        num_non_zero_datapoints = len(spectrum)

    if c.partition_sparsity:
        zeros_partitioned = np.array_split(np.zeros(len(spectrum)), num_non_zero_datapoints)
        ones_partitioned = np.array_split(np.ones(num_non_zero_datapoints), num_non_zero_datapoints)
        a_partitioned = [set_ones(zeros, int(sum(ones))) for zeros, ones in zip(zeros_partitioned, ones_partitioned)]
        for a in a_partitioned:
            np.random.shuffle(a)
        a = np.concatenate(a_partitioned)
    else:
        a = np.zeros(len(spectrum))
        a[:num_non_zero_datapoints] = 1
        np.random.shuffle(a)

    incomplete_spectrum = list(np.multiply(a, np.array(spectrum)))
    non_zero_indices = np.nonzero(incomplete_spectrum)
    non_zero_values = [val for val in incomplete_spectrum if val]
    normalised_non_zero = spectrum_normalisation(non_zero_values)
    # It appears that here, the normalisation occurs on the masked values,
    # which is correct

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
    sampler=None,
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

    if sampler is True:
        print('Creating WeightedRandomSampler to balance distribution of sO2 labels in dataset')
        labels = []
        for data, label in iter(dataset):
            labels.append(np.array(label[0]))

        hist, _ = np.histogram(labels)
        weights = 1 / hist

        # Need to map weights back to samples
        samples_weight = np.array([weights[int(np.floor(label * 10))] for label in labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    # Separate if block from above to prevent needless recreation of sampler
    if sampler:
        print('Using WeightedRandomSampler to balance dataset')
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)  # N.B. sampler does shuffling
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, sampler


# Example calls of prepare_dataloader()
# training_dataloader = prepare_dataloader(c.data_path, c.experiment_name, 'training', c.allowed_datapoints, c.batch_size)
# validation_dataloader = prepare_dataloader(c.data_path, c.experiment_name, 'validation', c.allowed_datapoints, c.batch_size)
# test_dataloader = prepare_dataloader(c.data_path, c.experiment_name, 'test', c.allowed_datapoints, c.batch_size)
