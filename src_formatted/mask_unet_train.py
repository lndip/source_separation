import numpy as np
import torch 


from torch import no_grad
from torch.optim import Adam
from torch.utils.data import random_split
from pathlib import Path

from getting_and_init_the_data import get_dataset, get_data_loader
from mask_unet import MaskUnet
from utils import PICKLE_DATA, save_masking_state_dict
from parameters import *


def forward_backward_pass_train(mask_unet,
                          dataloader,
                          optimizer,
                          device):

    losses = []

    if optimizer is not None:
        mask_unet.train()
    else:
        mask_unet.eval()

    for batch in dataloader:
        if optimizer is not None:
            optimizer.zero_grad()
        
        # Get the batches
        x, y = batch

        # Pass the data to the appropriate device.
        x = x.float().to(device)
        y = y.to(device)

        # Reshape the audio to (batch_size, 1, _, _)
        x = torch.unsqueeze(x, dim=1)
        y = torch.unsqueeze(y, dim=1)

        # Get the predictions of our model.
        mask = mask_unet(x)
        y_hat = torch.mul(x, mask)

        # Calculate the loss of our model.
        loss = torch.nn.functional.l1_loss(input=y_hat, target=y)

        # Back prppagation
        if optimizer is not None:
            loss.backward()

            optimizer.step()

        # Log the loss of the batch
        losses.append(loss.item())

    return mask_unet, np.mean(losses)


def train_masking_network(batch_size,
                        patience,
                        job_idx,
                        device,
                        epochs=5000):
    
    #Create the network
    mask_unet = MaskUnet(conv1_output_dim = conv1_output_dim,
                        conv2_output_dim = conv2_output_dim,
                        conv3_output_dim = conv3_output_dim,
                        conv4_output_dim = conv4_output_dim,
                        conv5_output_dim = conv5_output_dim,
                        conv6_output_dim = conv6_output_dim,
                        conv1_kernel_size = conv1_kernel_size,
                        conv2_kernel_size = conv2_kernel_size,
                        conv3_kernel_size = conv3_kernel_size,
                        conv4_kernel_size = conv4_kernel_size,
                        conv5_kernel_size = conv5_kernel_size,
                        conv6_kernel_size = conv6_kernel_size,
                        dropout = dropout,
                        conv1_stride = conv1_stride,
                        conv2_stride = conv2_stride,
                        conv3_stride = conv3_stride,
                        conv4_stride = conv4_stride,
                        conv5_stride = conv5_stride,
                        conv6_stride = conv6_stride,
                        conv1_padding = conv1_padding,
                        conv2_padding = conv2_padding,
                        conv3_padding = conv3_padding,
                        conv4_padding = conv4_padding,
                        conv5_padding = conv5_padding,
                        conv6_padding = conv6_padding)

    # Pass model to the available device.
    mask_unet = mask_unet.to(device) 

    # Give the parameters of Unet to an optimizer.
    optimizer = Adam(params=mask_unet.parameters(), lr=1e-3)

    # Getting and initializing the data
    train_dataset = get_dataset("train", PICKLE_DATA)
    val_dataset = get_dataset("val", PICKLE_DATA)
    train_dataloader = get_data_loader(train_dataset, batch_size, True, False)
    val_dataloader = get_data_loader(val_dataset, batch_size, False, False)

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0

    # Epoch metrics
    epoch_train_loss = []
    epoch_val_loss = []

    # Start training
    for epoch in range(epochs):
        mask_unet, train_loss = forward_backward_pass_train(mask_unet, 
                                                train_dataloader, 
                                                optimizer=optimizer, 
                                                device=device)

        with no_grad():
            mask_unet, val_loss = forward_backward_pass_train(mask_unet, 
                                                        val_dataloader, 
                                                        optimizer=None, 
                                                        device=device)
        
        epoch_train_loss.append(train_loss)
        epoch_val_loss.append(val_loss)

        print(f'Epoch: {epoch:03d} | '
        f'Mean training loss: {train_loss:7.4f} | '
        f'Mean validation loss {val_loss:7.4f}')

        # Early stopping
        if val_loss < lowest_validation_loss:
            lowest_validation_loss = val_loss
            best_validation_epoch = epoch
            torch.save(mask_unet.state_dict(), Path("masking_network","best_mask", f"#{job_idx}", "best_mask.pt"))

        if epoch - best_validation_epoch > patience:
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            break

    # Load the best model into the mask
    mask_unet.load_state_dict(torch.load(Path("masking_network","best_mask", f"#{job_idx}", "best_mask.pt")))
    # Save the model's state dict
    mask_file_path = save_masking_state_dict(mask_unet, job_idx)

    return epoch_train_loss, epoch_val_loss, mask_file_path

