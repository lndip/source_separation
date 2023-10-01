from typing import List, Union
import os
import pathlib
from pathlib import Path
from mask_unet import MaskUnet
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Global constants
MIXTURES = Path("source_separation", "mask_data", "mixtures")
SOUND_EVENTS = Path("source_separation", "mask_data", "targets")
PICKLE_DATA = Path("source_separation", "pickle_data")
MASKING_NET_DIR = Path("source_separation", "model")
TEST_RESULTS = Path("source_separation", "test_result")


def get_files_from_dir_with_os(dir_name: str) \
        -> List[str]:
    """Returns the files in the directory `dir_name` using the os package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[str]
    """
    return os.listdir(dir_name)


def get_files_from_dir_with_pathlib(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.\
        including the path of the parent directories.
    :rtype: list[pathlib.Path]
    """
    return list(pathlib.Path(dir_name).iterdir())
    

def empty_dir(path: str) \
        -> None:
    """Empty a directory 
    :param path: Path to the directory
    :type path:str
    """
    files = os.listdir(path)
    for file in files:
        os.remove(os.path.join(path, file))


def save_masking_state_dict(masking_net: MaskUnet, job_idx) -> None:
    """Save the masking network state dicts to its directory.

    :param masking_net: The masking network
    :type masking_net: MaskUnet
    """
    mask_file_name = 'mask_unet-' + str(job_idx) + '.pt'
    mask_file_path = Path(MASKING_NET_DIR, mask_file_name)

    torch.save(masking_net.state_dict(), mask_file_path)
    print('Masking network saved in ', mask_file_path)

    return mask_file_path


def save_spectrograms(mixture: np.ndarray,
                      speech: np.ndarray,
                      masked: np.ndarray,
                      path:str) -> None:
    figure, axis = plt.subplots(3)

    axis[0].set_title("mixture")
    mixture_img = librosa.display.specshow(mixture,
                                        ax=axis[0],
                                        sr=44100, 
                                        hop_length=441, 
                                        n_fft=1411,
                                        y_axis='log')
    figure.colorbar(mixture_img, ax=axis[0], format="%+2.f dB")

    axis[1].set_title("speech")
    speech_img = librosa.display.specshow(speech,
                                        ax=axis[1],
                                        sr=44100, 
                                        hop_length=441, 
                                        n_fft=1411,
                                        y_axis='log')
    figure.colorbar(speech_img, ax=axis[1], format="%+2.f dB")

    axis[2].set_title("masked")
    masked_img = librosa.display.specshow(masked,
                                        ax=axis[2],
                                        sr=44100,
                                        hop_length=441,
                                        n_fft=1411,
                                        y_axis='log')
    figure.colorbar(masked_img, ax=axis[2], format="%+2.f dB")

    figure.tight_layout()
    plt.savefig(path)
    plt.close()


def reconstruct_audio(stft: np.ndarray) -> np.ndarray:
    reconstructed_audio = librosa.istft(stft,
                                   n_fft=1411,
                                   hop_length=441,
                                   window="hamming")
    return reconstructed_audio
