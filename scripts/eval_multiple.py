"""
This script allows multiple models to be evaluated consecutively.
The config options defined in config.py are altered between runs to vary the model parameters.
"""
import numpy as np
import pandas as pd
import matplotlib
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.eval import eval_model
from qPAI_cINN_uncertainty_estimation.viz import plot_abs_error_comparison, plot_rel_error_comparison

if __name__ == "__main__":

    matplotlib.use('Agg')

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

        for eval_partitioned_bool in [True, False]:

            c.partition_sparsity = eval_partitioned_bool
            eval_part = 'eval_part' if eval_partitioned_bool else 'eval_unpart'

            for short_name, experiment_name in experiment_names.items():

                for label, n_wavelengths in allowed_wavelengths.items():

                    c.experiment_name = experiment_name
                    c.allowed_datapoints = n_wavelengths
                    balanced = '_balanced' if c.balance_dataset else ''

                    model_name = f"{short_name}_{label}_{partitioned}{balanced}"

                    eval_sparsities = {'3': [3], '5': [5], '10': [10], '25': [25], '40': [40]}
                    for sparsity_label, sparsity in eval_sparsities.items():

                        print(f"\n\n========== EVALUATING {model_name} at {sparsity_label} wavelengths ({eval_part}) ==========\n")

                        df, calib_df, row = eval_model(
                            model_name=model_name,
                            experiment_name=experiment_name,
                            allowed_datapoints=sparsity,
                            eval_str=f'_{eval_part}_{sparsity_label}',
                        )

                        rows.append(row)

    final_df = pd.DataFrame(rows)
    output_csv = c.output_dir / 'multiple_eval.csv'
    final_df.to_csv(output_csv)
