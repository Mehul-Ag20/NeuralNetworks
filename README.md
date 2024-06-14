## README

### Overview

This repository contains multiple notebooks demonstrating the implementation and training of different neural network architectures on both image and audio datasets. The main architectures covered are ResNet, VGG, and an Inception module-based network. Additionally, it includes custom data loaders and a Feed-Forward neural network for the MNIST dataset.

### Files

- **CustomDataLoader.ipynb**: Contains code for downloading the MNIST dataset and creating custom data loaders using both `torch.utils.data.Dataset` and a scratch implementation. It also compares the performance of these data loaders across different batch sizes.
- **MNIST.ipynb**: Implements and trains a Feed-Forward neural network on the MNIST dataset using the most effective data loader identified from the previous notebook.
- **ResNet.ipynb**: Implements a ResNet architecture with 18 blocks and trains it on both image and audio datasets.
- **VGG.ipynb**: Implements a modified VGG architecture and trains it on both image and audio datasets.
- **Inception.ipynb**: Implements a network using inception modules and trains it on both image and audio datasets.

### Instructions

#### Custom Data Loader

1. **Download the MNIST Dataset**:
   The notebook contains the necessary code to download the MNIST dataset using PyTorch's `datasets` module.

2. **Create a Custom DataLoader Using PyTorch**:
   The notebook demonstrates how to create a custom data loader using `torch.utils.data.Dataset` and `DataLoader`.

3. **Create a Custom DataLoader from Scratch**:
   The notebook also includes a custom data loader implementation from scratch and compares its performance with the PyTorch data loader across different batch sizes (128, 256, 512, 1024).

4. **Performance Comparison**:
   The notebook plots the relationship between batch size and loading time for both data loaders.

#### MNIST Feed-Forward Neural Network

1. **Network Architecture**:
   The notebook implements a Feed-Forward neural network with four hidden layers, each comprising at least 32 neurons. The network is trained using ReLU activation functions, Cross-Entropy loss, and the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.0003.

2. **Training**:
   The network is trained for 60 epochs, and graphs depicting the loss and accuracy during training, validation, and testing are plotted.

#### ResNet

1. **Architecture**:
   Implements a ResNet architecture comprising 18 blocks. Each block contains a sequence of convolution, batch normalization, and ReLU activation.

2. **Training**:
   The ResNet is trained on both image and audio datasets. For image data, 2D convolutions and batch normalization are used. For audio data, 1D convolutions and batch normalization are used. Cross-Entropy is the loss function, and Adam is the optimizer.

#### VGG

1. **Architecture**:
   Implements a modified VGG architecture where after each pooling layer, the number of channels is reduced by 35%, and the kernel size is increased by 25%. The network is structured into blocks, with the nth block and mth layer denoted as Conv n-m.

2. **Training**:
   The VGG network is trained on both image and audio datasets using Cross-Entropy loss and the Adam optimizer.

#### Inception Module

1. **Architecture**:
   Implements a network comprising four inception blocks. Each inception block contains a sequence of convolution, batch normalization, and ReLU activation with n×n convolution filters.

2. **Training**:
   The modified inception network is trained on both image and audio datasets using the same loss function and optimizer as the previous architectures.

### File Names

- Custom DataLoader: `CustomDataLoader.ipynb`
- MNIST Feed-Forward Neural Network: `MNIST.ipynb`
- ResNet: `ResNET`
- VGG: `VGG`
- Inception Module: `Inception`