import torch
import tqdm
import numpy as np
from datetime import datetime
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.model import WrappedModel, save, load
from qPAI_cINN_uncertainty_estimation.data import prepare_dataloader
from qPAI_cINN_uncertainty_estimation.init_log import init_logger


if __name__ == "__main__":

    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    output_dir = c.output_dir / start_time
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = c.log_dir / f"test_{start_time}.log"
    logger = init_logger(log_file.resolve())

    test_dataloader = prepare_dataloader(
        c.data_path, c.experiment_name, 'test', c.allowed_datapoints, c.batch_size
    )

    model = WrappedModel()
    if c.use_cuda:
        model.cuda()
    optim = torch.optim.Adam(
        model.params_trainable,
        lr=c.lr,
        betas=c.adam_betas,
        eps=c.eps,
        weight_decay=c.weight_decay,
    )
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=c.gamma)

    saved_state_file = c.output_dir / c.load_date / f"{c.load_date}@cinn.pt"

    load(saved_state_file.resolve(), model, optim)

    model.eval()  # Optional when not using Model Specific layer
    for i_val, (data, label) in enumerate(test_dataloader):
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
