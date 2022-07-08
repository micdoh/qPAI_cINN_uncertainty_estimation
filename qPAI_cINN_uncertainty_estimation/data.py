import torch
from torch.utils.data import Dataset, DataLoader


def switch_seq_feat(tensor):
    # Return a view of the tesor with axes rearranged
    return torch.permute(tensor, (1, 0)).float()


def float_transform(tensor):
    return tensor.float()


class MultiSpectralPressureO2Dataset(Dataset):
    def __init__(self, spectra, oxygenations):
        self.data = spectra
        self.labels = oxygenations
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        # TODO - How can I add the input Gaussian to the dataloader? What does the input Gaussian look like? Would it be easier/preferable to input the outputs? Also, probably the Gaussian should be the same every time (?), such that the only information introduced is from the conditioning
        # TODO - Alternativel, can have the Gaussian within the Wrapped Model class and it gets entered on the forward pass.
        return data, label
