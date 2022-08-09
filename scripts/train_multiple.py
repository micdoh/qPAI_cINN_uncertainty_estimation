"""
This script allows multiple models to be trained consecutively.
The config options defined in config.py are altered between runs to vary the model parameters.
"""
import numpy as np
from qPAI_cINN_uncertainty_estimation.train import train

if __name__ == "__main__":

    allowed_wavelengths = {
        '2': [2],
        '5': [5],
        '10': [10],
        '15': [15],
        '20': [20],
        '25': [25],
        '30': [30],
        '35': [35],
        '40': [40],
        '41': [41],
        'flexi': np.arange(3, 42),
    }

    experiment_names = {
        'phantom': 'FlowPhantom_insilico_complicated',
        'generic': 'NoSkin_filtered',
        'melanin': 'Skin_filtered',
    }

    for short_name, experiment_name in experiment_names.items():

        for label, n_wavelengths in allowed_wavelengths.items():

            model_name = f"{short_name}_{label}"

            print(f"\n\n========== TRAINING {model_name} ==========\n")

            train(
                model_name=model_name,
                experiment_name=experiment_name,
                allowed_datapoints=n_wavelengths,
            )

