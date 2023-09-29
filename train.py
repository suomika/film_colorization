import os
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from dataloader import CustomImageDataset, data_to_images, images_to_data
from unet import UNet
from autoencoder import *
from utility import *


def train_loop(dataloader, model, loss_fn, optimizer):
    """

    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :return:
    """

    model.train()
    train_loss = 0

    for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(dataloader.dataset)
    print('\nTrain set: Average loss: {:.7f}'.format(train_loss))

    return train_loss


def validate(model, dataloader, loss_fn):
    """

    :param model:
    :param dataloader:
    :param loss_fn:
    :return:
    """

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            val_loss += loss_fn(output, target).item()

    val_loss /= len(dataloader.dataset)
    print('Validation set: Average loss: {:.7f}\n'.format(val_loss))

    return val_loss


def train(train_data, val_data, architecture, method, flow, lr, batch_size, gamma, save_model, epochs, model_path):
    """

    :param train_data:
    :param val_data:
    :param architecture:
    :param method:
    :param flow:
    :param lr:
    :param batch_size:
    :param gamma:
    :param save_model:
    :param epochs:
    :param model_path:
    :return:
    """

    print(f'Cuda available: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    model_folder = architecture + '_' + method + '_' + str(epochs) + '_epochs'
    folder = os.path.join(model_path, architecture)
    checkpoint_path = os.path.join(folder, model_folder)
    make_dir(checkpoint_path)
    print(f'Checkpoint path: {checkpoint_path}\n')

    n_input_channels = 6 if flow else 4

    if architecture == 'UNet':
        model = UNet(n_input_channels=n_input_channels)

    elif architecture == 'Autoencoder':
        model = Autoencoder(n_input_channels=n_input_channels)

    elif architecture == 'BigAutoencoder':
        model = BigAutoencoder(n_input_channels=n_input_channels)

    else: raise ValueError('This architecture is not defined!')

    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    train_loss, val_loss = [], []
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}\n-------------------------------')

        train_loss.append(train_loop(train_dataloader, model, loss_fn, optimizer))
        val_loss.append(validate(model, val_dataloader, loss_fn))
        scheduler.step()

        if epoch % 5 == 0 and save_model:
            filename = architecture + '_' + method + '_' + str(epoch) + '.pth'
            checkpoint(folder, model, os.path.join(model_folder, filename))
    print('Done!')

    return train_loss, val_loss