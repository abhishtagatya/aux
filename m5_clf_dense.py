import argparse
import os

from lib.dataset.audio_mnist import (
    AudioMNIST, RESAMPLE_16K_TO_8K, collate_fn
)
from lib.model.m5 import M5

import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm


def train(model, loader, epoch, log_interval):
    model.train()
    losses = []

    for batch_idx, (data, target) in enumerate(loader):

        data = data.to(device)
        target = target.to(device)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} ({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # record loss
        losses.append(loss.item())


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, loader, epoch):
    model.eval()
    correct = 0
    for data, target in loader:

        data = data.to(device)
        target = target.to(device)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)\n")


if __name__ == '__main__':

    # Get Script Argument
    parser = argparse.ArgumentParser(description="Wav2Vec2.0 ASR Inference")
    parser.add_argument("dirpath", help="Audio Dataset Folder", type=str)
    parser.add_argument("lr", help="Optimizer Learning Rate", type=str)
    parser.add_argument("weight_decay", help="Optimizer Weight Decay", type=str)
    parser.add_argument("scheduler_step", help="Scheduler Step Size", type=str)
    parser.add_argument("scheduler_gamma", help="Scheduler Gamma", type=str)
    parser.add_argument("epoch", help="Epoch", type=str)
    args = parser.parse_args()

    DIR = args.dirpath
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = AudioMNIST.load_from_dir(DIR, transform=RESAMPLE_16K_TO_8K, split=0.8)
    train_loader = train_ds.to_dataloader(batch_size=256, shuffle=True, collate=collate_fn, device=device)
    test_loader = test_ds.to_dataloader(batch_size=256, shuffle=False, collate=collate_fn, device=device)

    model = M5(1, 10)
    model.to(device)

    # Fine Tune the Parameters to get better results
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.scheduler_step), gamma=float(args.scheduler_gamma))

    log_interval = 20
    n_epoch = int(args.epoch)

    for epoch in range(n_epoch):
        train(model, train_loader, epoch, log_interval)
        test(model, train_loader, epoch)
        test(model, test_loader, epoch)
        scheduler.step()

    save_as = f"m5_10+{args.learning_rate}-{args.weight_decay}-sch{args.scheduler_gamma}.pth.tar"
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, save_as)
