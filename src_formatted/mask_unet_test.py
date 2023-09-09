import torch
import numpy as np
import librosa as lb
import soundfile as sf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from torch import no_grad, Tensor
from torchmetrics.functional.audio import signal_distortion_ratio
from pathlib import Path

from getting_and_init_the_data import get_dataset, get_data_loader
from mask_unet import MaskUnet
from utils import (PICKLE_DATA,
                   TEST_RESULTS,
                   reconstruct_audio,
                   save_spectrograms)
from parameters import *


def test_masking_network(device,
                         network_path,
                         batch_size,
                         result_dir_name,
                         mask_threshold=None):

    print("Getting network from: ", network_path, '\n\n')

    # Create directory to store testing results
    if not os.path.exists(os.path.join(TEST_RESULTS, result_dir_name)):
        os.mkdir(os.path.join(TEST_RESULTS, result_dir_name))
        os.mkdir(os.path.join(TEST_RESULTS, result_dir_name, "plot"))
        os.mkdir(os.path.join(TEST_RESULTS, result_dir_name, "audio"))
        os.mkdir(os.path.join(TEST_RESULTS, result_dir_name, "sdrs"))

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
    mask_unet.load_state_dict(torch.load(network_path, map_location=device))

    # Pass model to the available device.
    mask_unet = mask_unet.to(device)

    # Getting and initializing the data
    test_dataset = get_dataset('eval', PICKLE_DATA)
    test_dataloader = get_data_loader(test_dataset, batch_size, False, False)

    # Specify in evaluation mode
    mask_unet.eval()

    # Iteration
    i = 0
    sdrs = {}

    with no_grad():
        for batch in test_dataloader: 
            # Get the batch
            samples, event, speech = batch

            # Get the magnitude and the phase of x_test
            samples_mag = Tensor(np.abs(samples))
            samples_phase = lb.magphase(event.numpy())[1].squeeze() 
            # Get the magnitude of y_test
            event_mag = Tensor(np.abs(event)).unsqueeze(dim=1)
            speech_mag = Tensor(np.abs(speech))

            # Pass the data to the device and reshape x to (batch_size, 1, _, _)
            samples_mag = samples_mag.float().unsqueeze(dim=1).to(device)
            event_mag = event_mag.to(device)

            # Get the predictions of our model.
            mask = mask_unet(samples_mag)
            if mask_threshold is not None:
                mask = (mask >= mask_threshold).float()
            # print(mask)
            samples_masked = torch.mul(mask, samples_mag)

            loss = torch.nn.functional.l1_loss(input=samples_masked, target=event_mag)

            # Audio reconstruction
            # Create directory for the reconstructed audio
            audio_dir = os.path.join(TEST_RESULTS, result_dir_name, "audio", f"eval_{i}")
            if not os.path.exists(audio_dir):
                os.mkdir(audio_dir)

            # Get reconstructed audios
            event_mag = torch.squeeze(event_mag).numpy()
            samples_masked = torch.squeeze(samples_masked).numpy()

            samples_masked_reconstructed = reconstruct_audio(samples_masked*samples_phase)
            event_reconstructed = reconstruct_audio(event.squeeze().numpy())
            samples_reconstructed = reconstruct_audio(samples.squeeze().numpy())
            speech_reconstructed = reconstruct_audio(speech.squeeze().numpy())

            print(event.shape)

            # Save reconstructed audios
            sf.write(os.path.join(audio_dir, "masked_audio.wav"), 
                                  samples_masked_reconstructed,
                                  44100)
            sf.write(os.path.join(audio_dir, "event.wav"), 
                                  event_reconstructed,
                                  44100)
            sf.write(os.path.join(audio_dir, "mixture.wav"), 
                                  samples_reconstructed,
                                  44100)
            sf.write(os.path.join(audio_dir, "speech.wav"),
                                  speech_reconstructed,
                                  44100)

            # Calculate SDR metric
            sdr = signal_distortion_ratio(Tensor(samples_masked_reconstructed), 
                                          Tensor(event_reconstructed))
            sdrs[i] = sdr

            i += 1

            print(f'Sample: {i:03d} | '
            f'Loss: {loss:7.4f} | ')

            # Save the spectrograms
            save_spectrograms(samples_mag.squeeze().numpy(), speech_mag.squeeze().numpy(), samples_masked, 
                              Path(TEST_RESULTS, result_dir_name, "plot", f"eval_{i}"))

    # Save the results of SDR calculation
    with Path(TEST_RESULTS, result_dir_name, "sdrs", "sdr.txt").open('w') as file:
        sum = 0
        for key in sdrs.keys():
            sum += sdrs[key]
            file.write(f"Sample {key}: SDR {sdrs[key]}")
            file.write('\n')
        file.write(f"Mean sdr: {sum/len(sdrs)}")




