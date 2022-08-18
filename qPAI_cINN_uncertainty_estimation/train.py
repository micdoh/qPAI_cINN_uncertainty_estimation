import torch
import tqdm
import numpy as np
import time
import glob
from datetime import datetime
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.model import WrappedModel, save, load, init_model
from qPAI_cINN_uncertainty_estimation.data import prepare_dataloader
from qPAI_cINN_uncertainty_estimation.init_log import init_logger
from qPAI_cINN_uncertainty_estimation.monitoring import config_string


def save_losses(loss_type: str, losses: list, output_dir, name):
    losses_file = output_dir / f"{name}@{loss_type}.npy"
    losses = np.array(losses)
    if c.load_for_retraining:
        with open(losses_file.resolve(), "rb") as f:
            prev_losses = np.load(f)
            losses = np.append(prev_losses, losses)
    with open(losses_file.resolve(), "wb") as f:
        np.save(f, losses)


def get_last_epoch(output_dir):
    file_search = output_dir / '*@loss_epoch*.npy'
    files = glob.glob(str(file_search))
    epoch_indices = [int(file.split('_')[-1].split('.')[0]) for file in files]
    epoch_indices.sort()
    return epoch_indices[-1]


def train(
        model_name=None,
        allowed_datapoints=c.allowed_datapoints,
        experiment_name=c.experiment_name
):

    model, optim, weight_scheduler = init_model()

    if c.load_for_retraining:
        saved_state_file = c.output_dir / c.load_date / f"{c.load_date}@cinn.pt"
        load(saved_state_file.resolve(), model, optim)
        optim.param_groups[0]['capturable'] = True  # Needs to be set to allow reloading for training
        start_time = c.load_date

    else:
        start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    start = time.time()
    name = model_name if model_name else start_time

    output_dir = c.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = c.log_dir / f"training_{name}.log"
    logger = init_logger(log_file.resolve(), log_file.stem)

    config_details_file = output_dir / f"{name}@train_config.txt"
    logger.info(config_string(config_details_file.resolve()))

    min_valid_loss = np.inf

    try:

        epoch_losses = []
        valid_losses = []
        i_epoch = get_last_epoch(output_dir) if c.load_for_retraining else 0
        no_improvement_epochs = 0
        sampler = True

        while True:

            i_epoch += 1

            # N.B. initialising the dataloaders afresh with every epoch allows the masking to be re-performed
            training_dataloader, sampler = prepare_dataloader(
                c.data_path, experiment_name, "training", allowed_datapoints, c.batch_size, sampler=sampler
            )
            validation_dataloader, _ = prepare_dataloader(
                c.data_path, experiment_name, 'validation', allowed_datapoints, c.batch_size
            )

            loss_history = []

            iterator = tqdm.tqdm(
                iter(training_dataloader),
                total=len(training_dataloader),
                leave=True,
                position=0,
                mininterval=1.0,
                ncols=83,
            )

            for i, (data, label) in enumerate(iterator):
                # Send data to GPU or CPU (https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)
                data, label = data.to(c.device), label.to(c.device)
                # pass to INN and get transformed variable z and log Jacobian determinant
                z, log_jac_det = model(data, label)
                # calculate the negative log-likelihood of the model with a standard normal prior
                nll = (
                    torch.mean(z**2) / 2 - torch.mean(log_jac_det) / c.total_data_dims
                )
                # TODO - Could implement bidirectional training as done here:
                #  (https://github.com/VLL-HD/analyzing_inverse_problems/blob/master/inverse_problems_science/train.py)
                # backpropagate and update the weights
                nll.backward()
                if c.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(model.params_trainable, 10.0)
                optim.step()
                optim.zero_grad()
                loss_history.append(nll.item())

            loss_file = output_dir / f"{name}@loss_epoch_{i_epoch}.npy"
            with open(loss_file.resolve(), "wb") as f:
                np.save(f, np.array([loss_history]))

            epoch_loss = np.mean(np.array(loss_history), axis=0)
            epoch_losses.append(epoch_loss)

            if i_epoch > 0 and (i_epoch % c.checkpoint_save_interval) == 0:
                model_checkpoint_file = output_dir / f"{name}@cinn_checkpoint_{i_epoch / c.checkpoint_save_interval:.1f}"
                save(
                    model_checkpoint_file.resolve(),
                    optim,
                    model,
                )

            valid_loss = []
            model.eval()  # Optional when not using Model Specific layer
            for i_val, (data, label) in enumerate(validation_dataloader):
                # Transfer Data to GPU if available
                data, label = data.to(c.device), label.to(c.device)
                # Forward Pass
                z, log_jac_det = model(data, label)
                # Find the Loss
                nll = (
                        torch.mean(z ** 2) / 2 - torch.mean(log_jac_det) / c.total_data_dims
                )
                valid_loss.append(nll.item())

            valid_loss = np.mean(np.array(valid_loss), axis=0)
            logger.info(f'Model: {name} \t Epoch {i_epoch} \t '
                  f'Training Loss: {epoch_loss:.6f} \t '
                  f'Validation Loss: {valid_loss:.6f}')
            model.train()

            if min_valid_loss > valid_loss:
                logger.info(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                model_file = output_dir / f"{name}@cinn.pt"
                save(model_file.resolve(), optim, model)
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            valid_losses.append(valid_loss)

            weight_scheduler.step()

            # End training
            if (no_improvement_epochs > c.no_improvement_epoch_cutoff
                or i_epoch > c.max_epochs) \
                    and i_epoch > c.min_epochs :
                break

    except Exception as e:
        model_abort_file = output_dir / f"{name}@cinn_ABORT.pt"
        save(model_abort_file.resolve(), optim, model)
        logger.error(str(e))
        raise e

    finally:  # Always save loss data

        save_losses('epoch_losses', epoch_losses, output_dir, name)

        save_losses('valid_losses', valid_losses, output_dir, name)

        logger.info(f"\n--- Min. validation loss = {min_valid_loss:.4f} ---\n")

        logger.info("--- %s seconds ---" % (time.time() - start))


if __name__ == "__main__":
    train()
