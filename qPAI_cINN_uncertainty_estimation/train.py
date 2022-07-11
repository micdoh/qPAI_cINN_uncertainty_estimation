import torch
import tqdm
import torch.nn as nn
import numpy as np
from datetime import datetime
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.model import WrappedModel, save
from qPAI_cINN_uncertainty_estimation.data import prepare_dataloader


if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    model = WrappedModel()
    if c.use_cuda:
        model.cuda()
        model = nn.DataParallel(model)

    training_dataloader = prepare_dataloader(c.data_path, c.experiment_name, 'training', c.allowed_datapoints, c.batch_size)

    optim = torch.optim.Adam(model.params_trainable, lr=c.lr, betas=c.adam_betas, eps=c.eps, weight_decay=c.weight_decay)
    #weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=c.gamma)

    try:
        for i_epoch in range(c.n_epochs):

            loss_history = []

            iterator = tqdm.tqdm(
                iter(training_dataloader),
                total=len(training_dataloader),
                leave=False,
                mininterval=1.,
                ncols=83
            )

            for i, (data, label) in enumerate(iterator):
                # Send data to GPU or CPU (https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)
                data, label = data.to(c.device), label.to(c.device)
                # pass to INN and get transformed variable z and log Jacobian determinant
                z, log_jac_det = model(data, label)
                # calculate the negative log-likelihood of the model with a standard normal prior
                nll = torch.mean(z ** 2) / 2 - torch.mean(log_jac_det) / c.total_data_dims
                # backpropagate and update the weights
                nll.backward()
                torch.nn.utils.clip_grad_norm_(model.params_trainable, 10.)
                optim.step()
                optim.zero_grad()
                loss_history.append(nll.item())

            with open(f'loss_epoch_{i_epoch}_{timestamp}.npy', 'wb') as f:
                np.save(f, np.array([loss_history]))

            epoch_losses = np.mean(np.array(loss_history), axis=0)

            if i_epoch > 0 and (i_epoch % c.checkpoint_save_interval) == 0:
                model.save(f"{c.output_file}_{timestamp}_checkpoint_{i_epoch / c.checkpoint_save_overwrite:.1f}")

    except Exception as e:
        save(f"{c.output_file}_{timestamp}_ABORT.pt", optim, model)
        raise e

save(f"{c.output_file}_{timestamp}.pt", optim, model)
