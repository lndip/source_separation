from typing import Union, MutableMapping, Optional
from pathlib import Path
from pickle import dump
import librosa as lb
import numpy as np
import os
from utils import get_files_from_dir_with_os, empty_dir, MIXTURES, SOUND_EVENTS, PICKLE_DATA

__docformat__ = 'reStructuredText'


def stft_transform(audio_data: np.ndarray,
                    n_fft: Optional[int] = 1411,
                    hop_length: Optional[int] = 441) \
        -> np.ndarray:
    """Apply STFT on the audio data in time domain

    :param audio_file: Audio file in time domain.
    :type audio_file: np.ndarray
    :param sr: Sampling frequency of audio file, defaults to 44100.
    :type sr: Optional[int]
    :param n_fft: STFT window length (in samples), defaults to 1024.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 320.
    :type hop_length: Optional[int]
    :return: Augio signal in time- frequency domain
    :rtype: np.ndarray
    """
    stft = lb.stft(y=audio_data,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window='hamming')
    return stft


def serialize_features_and_classes(
        pickle_path: Union[Path, str],
        features_and_classes: MutableMapping[str, Union[np.ndarray, int]]) \
        -> None:
    """Serializes the features and classes.

    :param pickle_path: Path of the pickle file
    :type pickle _path: Path|str
    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    """
    with pickle_path.open('wb') as pickle_file:
        dump(features_and_classes, pickle_file)


def handle_one_data_pair(split: str,
                        mixture_path: Union[Path, str],
                        target_path: Union[Path, str],
                        output_pickle_path: Union[Path, str]) \
        -> None:
    """
    Read data from time domain, convert it to time-frequency domain
    Create features_and_classes dict for the data pair and save it in pickle file.

    :param mixture_path: The mixture data path
    :type mixture_path: Path|str
    :param target_path: The target data path
    :type target_path: Path|str
    :param output_pickle_path: The pickle file path
    :type output_pickle_path: Path|str
    """
    features_and_classes = {}

    # get the audio data (time domain)
    mixture_data = np.load(mixture_path)
    target_data = np.load(target_path)
    speech_data = mixture_data - target_data

    mixture_spec = None
    target_spec = None
    speech_spec = None

    if split == 'train' or split == 'val':
        mixture_spec = np.abs(stft_transform(mixture_data))
        target_spec = np.abs(stft_transform(target_data))
        speech_spec = np.abs(stft_transform(speech_data))

    elif split == 'eval':
        mixture_spec = stft_transform(mixture_data)
        target_spec = stft_transform(target_data)
        speech_spec = stft_transform(speech_data)

    # assign values to the dict
    features_and_classes["source"] = mixture_spec
    features_and_classes["target"] = target_spec
    features_and_classes["speech"] = speech_spec

    serialize_features_and_classes(Path(output_pickle_path), features_and_classes)


def create_pickle_data(split: str) -> None:
    """Create pickle data for the chosen split set

    :param split: The dataset split
    :type split: str
    """
    # Get the split's directory path
    mixtures_dir = Path.joinpath(Path(MIXTURES), Path(split))
    targets_dir =  Path.joinpath(Path(SOUND_EVENTS), "eval") if split == "eval" else Path.joinpath(Path(SOUND_EVENTS), "dev") 
    pickle_dir = Path.joinpath(Path(PICKLE_DATA), Path(split))

    empty_dir(pickle_dir)

    # Loop over the mixtures and create pickle data
    for mixture in get_files_from_dir_with_os(mixtures_dir):
        sound_event = mixture.split('_')[0] + '_' + mixture.split('_')[1] + '_' + mixture.split('_')[2] + '.npy'

        mixture_path = os.path.join(mixtures_dir, mixture)
        target_path = os.path.join(targets_dir, sound_event)
        output_pickle_path = os.path.join(pickle_dir, f'{mixture}.pickle')
        handle_one_data_pair(split, mixture_path, target_path, output_pickle_path)
        

def main():
    for split in ["train", "val", "eval"]:
        create_pickle_data(split)


if __name__ == "__main__":
    main()