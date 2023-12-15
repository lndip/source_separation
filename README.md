# Source separation neural network
This is the implementation of the source separation neural network which employs the U-net structure. The implementation refers to [Singing Voice Separation with Deep U-Net Convolutional Networks](https://openaccess.city.ac.uk/id/eprint/19289/1/).

# Model architecture
The encoder and decoder blocks of the U-net each contains 6 convolutional blocks. The model takes the STFT magnitude spectrogram of the input signal and outputs masked STFT spectrogram.

# Folder structure
```
source_separation
+--README.md
+--mask_data
|  +--mixtures
|     +--train
|     +--val
|     +--test
|  +--targets
|     +--train
|     +--val
|     +--test
+--model
+--pickle_data
|  +--train
|  +--val
|  +--test
+--src_formatted
+--test_result
```

# Installation
To run the code, `python`, `pytorch`, `torchaudio`, `numpy`, and `librosa` are required.

# How to run
1. Have the data in the `mask_data` folder as the structure above. Every sample in training and validation set must of equal length for batch processing.
2. Run `serialize.py` to obtain the pickle data.
3. Run `mask_main.py` to execute the training and inference.

# Authors
Diep Luong


