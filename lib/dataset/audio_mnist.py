import glob
import random

import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader

RESAMPLE_16K_TO_8K = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    tensors, target = [], []
    for waveform, label in batch:
        tensors += [waveform]
        target += [label]

    tensors = pad_sequence(tensors)
    target = torch.stack(target)

    return tensors, target


class AudioMNIST(Dataset):

    def __init__(self, data, transform=None):
        super(AudioMNIST, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    @classmethod
    def load_from_dir(cls, data_dir, transform=None, split=0, shuffle=True):
        dir_list = glob.glob(data_dir)

        if shuffle:
            random.shuffle(dir_list)

        if 0 < split < 1:
            len_split = int(len(dir_list) * split)
            train, test = dir_list[0:len_split], dir_list[len_split:]
            return cls(train, transform=transform), cls(test, transform=transform)

        return cls(dir_list, transform=transform)

    def to_dataloader(self, batch_size=256, shuffle=True, collate=None, device=None):
        if device == "cuda":
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=collate,
                          num_workers=num_workers,
                          pin_memory=pin_memory)

    def __getitem__(self, idx):
        label = self.data[idx].split('/')[-1].split('_')[0]
        waveform, sample_rate = torchaudio.load(self.data[idx])

        if self.transform:
            waveform = self.transform(waveform)
        return waveform, torch.tensor(int(label))
