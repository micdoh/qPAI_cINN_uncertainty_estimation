"""
This script allows multiple models to be trained consecutively.
The config options defined in config.py are altered between runs to vary the model parameters.
"""
import numpy as np
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.train import train

if __name__ == "__main__":

    allowed_wavelengths = {
        #'3': [3],
        #'5': [5],
        #'10': [10],
        #'25': [25],
        '40': [40],
        #'flexi': np.arange(3, 42),
    }

    experiment_names = {
        #'phantom': 'FlowPhantom_insilico_complicated',
        'generic': 'NoSkin_filtered',
        #: 'Skin_filtered',
    }

    for short_name, experiment_name in experiment_names.items():

        for label, n_wavelengths in allowed_wavelengths.items():

            c.experiment_name = experiment_name
            c.allowed_datapoints = n_wavelengths

            partitioned = 'partitioned' if c.partition_sparsity else 'unpartitioned'
            balanced = '_balanced' if c.balance_dataset else ''

            model_name = f"{short_name}_{label}_{partitioned}{balanced}"

            print(f"\n\n========== TRAINING {model_name} ==========\n")

            train(
                model_name=model_name,
                experiment_name=experiment_name,
                allowed_datapoints=n_wavelengths,
            )
