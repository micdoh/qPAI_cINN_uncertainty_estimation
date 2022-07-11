import torch
import tqdm
import torch.nn as nn
import numpy as np
from datetime import datetime
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.model import WrappedModel, save
from qPAI_cINN_uncertainty_estimation.data import prepare_dataloader
from qPAI_cINN_uncertainty_estimation.init_log import init_logger


if __name__ == "__main__":

    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    log_file = c.log_dir / f"training_{start_time}.log"
    logger = init_logger(log_file.resolve())

    model = WrappedModel()
    if c.use_cuda:
        model.cuda()
        # model = nn.DataParallel(model)

    training_dataloader = prepare_dataloader(
        c.data_path, c.experiment_name, "training", c.allowed_datapoints, c.batch_size
    )
    validation_dataloader = prepare_dataloader(
        c.data_path, c.experiment_name, 'validation', c.allowed_datapoints, c.batch_size
    )

    optim = torch.optim.Adam(
        model.params_trainable,
        lr=c.lr,
        betas=c.adam_betas,
        eps=c.eps,
        weight_decay=c.weight_decay,
    )
    # weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=c.gamma)
    min_valid_loss = np.inf

    try:

        epoch_losses = []

        for i_epoch in range(c.n_epochs):

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
                #data, label = data.to(c.device), label.to(c.device)
                if c.use_cuda:
                    data, label = data.cuda(), label.cuda()
                # pass to INN and get transformed variable z and log Jacobian determinant
                z, log_jac_det = model(data, label)
                # calculate the negative log-likelihood of the model with a standard normal prior
                nll = (
                    torch.mean(z**2) / 2 - torch.mean(log_jac_det) / c.total_data_dims
                )
                # backpropagate and update the weights
                nll.backward()
                torch.nn.utils.clip_grad_norm_(model.params_trainable, 10.0)
                optim.step()
                optim.zero_grad()
                loss_history.append(nll.item())

            loss_file = c.output_dir / f"loss_epoch_{i_epoch}_{start_time}.npy"
            with open(loss_file.resolve(), "wb") as f:
                np.save(f, np.array([loss_history]))

            epoch_loss = np.mean(np.array(loss_history), axis=0)
            epoch_losses.append(epoch_loss)

            logger.info(f"Epoch {i_epoch} \t\t Training Loss: {epoch_loss}")

            if i_epoch > 0 and (i_epoch % c.checkpoint_save_interval) == 0:
                save(
                    f"{c.output_file}_{start_time}_checkpoint_{i_epoch / c.checkpoint_save_interval:.1f}",
                    optim,
                    model,
                )

            valid_loss = []
            model.eval()  # Optional when not using Model Specific layer
            for i_val, (data, label) in enumerate(validation_dataloader):
                # Transfer Data to GPU if available
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                # Forward Pass
                z, log_jac_det = model(data, label)
                # Find the Loss
                nll = (
                        torch.mean(z ** 2) / 2 - torch.mean(log_jac_det) / c.total_data_dims
                )
                # Calculate Loss
                valid_loss.append(nll.item())

            valid_loss = np.mean(np.array(valid_loss), axis=0)
            logger.info(f'Epoch {i_epoch} \t\t '
                  f'Training Loss: {epoch_loss} \t\t '
                  f'Validation Loss: {valid_loss / len(validation_dataloader)}')
            model.train()

            if abs(min_valid_loss) > abs(valid_loss):
                logger.info(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                save(f"{c.output_file}_{start_time}.pt", optim, model)

    except Exception as e:
        save(f"{c.output_file}_{start_time}_ABORT.pt", optim, model)
        raise e

epoch_losses_file = c.output_dir / f"epoch_losses_{start_time}.npy"
with open(epoch_losses_file.resolve(), "wb") as f:
    np.save(f, np.array([epoch_losses]))

#save(f"{c.output_file}_{start_time}.pt", optim, model)
