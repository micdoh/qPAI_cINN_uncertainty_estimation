import torch
import random
import numpy as np


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
    for index in non_zero_indices[0]:
        incomplete_spectrum[index] = normalised_non_zero[i]
        i += 1

    normalised_incomplete_spectrum = np.array(incomplete_spectrum)

    return normalised_incomplete_spectrum


def batch_spectrum_processing(batch, allowed_datapoints):
    processed = []

    for spectrum in batch:

        processed.append(spectrum_processing(spectrum, allowed_datapoints))
    return torch.tensor(np.array(processed))
