# Source separation neural network
This is the implementation of the source separation neural network which employs the U-net structure

# Model architecture
The encoder and decoder blocks both contains 6 convolutional blocks. The model takes the STFT magnitude spectrogram of the input signal and outputs masked STFT spectrogram.

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


