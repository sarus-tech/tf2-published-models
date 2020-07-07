# Sarus published models

Sarus implementation of classical ML models. The models are implemented using the Keras API of tensorflow 2. Vizualization are implemented and can be seen in tensorboard.

The required packages are managed with `pipenv` and can be installed using `pipenv install`. Please see the [pipenv documentation](https://pipenv-fork.readthedocs.io/en/latest/) for more information.

## Philosophy

These models' implementations are intended to be easy to read and to adapt by making use of the latest Tensorflow 2 library and Keras API.

## Basic usage

To install and train a model.

```shell
pipenv install
pipenv shell
python train.py
```

To visualize losses and reconstructions.

```shell
tensorboard --logdir ./logs/
```

## Available models

* [Simple Autoencoder](./simple_autoencoder)
* [Variational Autoencoder (VAE)](./vae)
* [Vector Quantized Autoencoder (VQ-VAE)](./vqvae)
* [PixelCNN](./pixelcnn)
* [Gated PixelCNN](./gated_pixelcnn)
* [PixelCNN++](./pixelcnn_plus)
* [Conditional Neural Processes](./cnp)
* [PixelSNAIL](./pixelsnail)
