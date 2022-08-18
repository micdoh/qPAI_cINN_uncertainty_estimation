import pandas as pd
import numpy as np
import pathlib

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

    print("Loading data...[DONE]")
    return (wavelengths, oxygenations, spectra, melanin_concentration,
            background_oxygenation, distances, depths, timesteps, tumour_mask, reference_mask,
            mouse_body_mask, background_mask)

if __name__ == '__main__':

    root = pathlib.Path(r'I:\research\seblab\data\group_folders\Janek\learned_pa_oximetry\validation_data')
    flow_1 = root / 'in_vitro' / 'Phantom1_flow_phantom_medium_melanin' / 'Phantom1_flow_phantom_medium_melanin.npz'
    flow_2 = root / 'in_vitro' / 'Phantom2_flow_phantom_medium_melanin' / 'Phantom2_flow_phantom_medium_melanin.npz'
