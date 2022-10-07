# Uncertainty-Aware Photoacoustic Oximetry using Conditional Invertible Neural Networks

This repository contains the scripts and functionality to define, train, and evaluate a conditional invertible neural network (cINN) for the purpose of uncertainty-aware quantitative photoacoustic imaging (qPAI), specifically pixel-wise oxygen estimation from multispectral images.

The code is arranged into the following subdirectories:

- `\notebooks` contains jupyter notebooks used to test and develop the main functionality.
- `\qPAI_cINN_uncertainty_estimation` is the module containing the model definition and training routines (and `config.py` configuration file).
- `\scripts` contains scripts to train models, evaluate performance, and plot results.
- `\tests` to test functionality (to be implemented).

____

## Quick setup

### Additional directories

The following additional directories must be created in the project root directory in order to execute project scripts successfully:

- `datasets` (to contain the train/eval/test datasets)
- `output` (to output the results of training and evaluation, inc. charts, dataframes, and trained models)
- `logs` (to output log files from each script run)

### Environment setup

In order to run the scripts, dependencies must be installed using pip *or* a virtual environment can be set up.

All project dependencies are defined in the `pyproject.toml`

#### Installing dependencies

From within the project root directory, run:

`python -m pip install .`

#### Virtual environment setup (preferred)

The virtual environment can be setup using either poetry or virtualenv/venv. Poetry is the preferred package manager. Installation instructions are found here: https://python-poetry.org/docs/

Once installed, poetry can be used to install dependencies or establish a virtual environment.

To install dependencies: `poetry install`

To establish virtual environment: `poetry shell`

___

## Configuration

Options for model parameters and training hyperparameters are defined in `config.py`. The values defined in config.py are imported by the rest of the module. These values are also edited in scripts, e.g. `train_multiple.py`, to allow programmatic updates to parameter values, e.g. between training runs.

In addition to parameters, `config.py` also sets the root paths for reading datasets and outputting any files.

The input/output paths vary depending on whether the machine on which the script is executing is CUDA-compatible (i.e. Linux server with GPU) or not (e.g. Windows laptop). Please check the paths are configured as you desire in `config.py`.

Other parameters/hyperparameters are labelled accordingly.
