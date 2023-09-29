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


def prediction(test_data, model_path, architecture, method, epochs, flow, batch_size, num):
    """

    :param test_data:
    :param model_path:
    :param architecture:
    :param method:
    :param epochs:
    :param flow:
    :param batch_size:
    :param num:
    :return:
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_input_channels = 6 if flow else 4

    if architecture == 'UNet':
        model = UNet(n_input_channels=n_input_channels)

    elif architecture == 'Autoencoder':
        model = Autoencoder(n_input_channels=n_input_channels)

    elif architecture == 'BigAutoencoder':
        model = BigAutoencoder(n_input_channels=n_input_channels)

    else: raise ValueError('This architecture is not defined!')

    model.to(device)

    model_folder = architecture + '_' + method + '_' + str(epochs) + '_epochs'
    folder = os.path.join(model_path, architecture)
    folder = os.path.join(folder, model_folder)
    filename = architecture + '_' + method + '_' + str(epochs) + '.pth'

    resume(model, folder, filename)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    test_features, test_labels = next(iter(test_dataloader))
    # print(f'Test feature batch shape: {test_features.size()}')
    # print(f'Test labels batch shape: {test_labels.size()}')

    prediction = model(test_features.to(device))
    prediction = prediction.cpu()
    prediction = prediction.detach()

    col_img, flow_img, grey_img, pred_img = data_to_images(test_features[num], prediction[num], use_flow=flow,
                                                           input_only=False)

    return col_img, flow_img, grey_img, pred_img


def plot_n_predictions(test_data, model_path, architecture, method, epochs, flow, batch_size, n):
    """

    :param test_data:
    :param model_path:
    :param architecture:
    :param method:
    :param epochs:
    :param flow:
    :param batch_size:
    :param n:
    :return:
    """

    for i in range(n):
        col_img, flow_img, grey_img, pred_img = prediction(test_data, model_path, architecture, method, epochs, flow,
                                                           batch_size, i)

        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[0].imshow(col_img)
        axs[0].axis('off')
        axs[0].set_title('Colored Previous Frame')

        axs[1].imshow(grey_img, cmap='gray')
        axs[1].axis('off')
        axs[1].set_title('Greyscale image')

        axs[2].imshow(pred_img)
        axs[2].axis('off')
        axs[2].set_title('Prediction')

        plt.tight_layout()
        plt.show()