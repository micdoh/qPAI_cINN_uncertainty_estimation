"""
This script allows multiple models to be evaluated consecutively.
The config options defined in config.py are altered between runs to vary the model parameters.
"""
import numpy as np
import pandas as pd
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.viz import (
    plot_training_epoch_losses,
    plot_validation_epoch_losses,
    plot_training_batch_losses,
)

if __name__ == "__main__":

    allowed_wavelengths = {
        '3': [3],
        '5': [5],
        '10': [10],
        '25': [25],
        '40': [40],
        'flexi': np.arange(3, 42),
    }

    experiment_names = {
        'phantom': 'FlowPhantom_insilico_complicated',
        'generic': 'NoSkin_filtered',
        'melanin': 'Skin_filtered',
    }

    rows = []

    for partitioned in ['partitioned', 'unpartitioned']:

        for short_name, experiment_name in experiment_names.items():

            for label, n_wavelengths in allowed_wavelengths.items():

                c.experiment_name = experiment_name
                c.allowed_datapoints = n_wavelengths

                model_name = f"{short_name}_{label}_{partitioned}"

                print(f"\n\n========== PLOTTING LOSSES for {model_name} ==========\n")

                output_dir = c.output_dir / model_name

                plot_training_batch_losses(model_name, dir=output_dir)
                plot_training_epoch_losses(model_name, dir=output_dir)
                plot_validation_epoch_losses(model_name, dir=output_dir)
