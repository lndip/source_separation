# Source separation neural network
This is the implementation of the source separation neural network which employs the U-net structure

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
<ol>
  <li>Have the data in the `mask_data` folder as the structure above. Every sample in training and validation set must of equal length for batch processing.</li>
  <li>Run `serialize.py` to obtain the pickle data.</li>
  <li>Run `mask_main.py` to execute the training and inference.</li>
</ol>

# Authors
Diep Luong


