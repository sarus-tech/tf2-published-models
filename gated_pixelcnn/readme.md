# Gated PixelCNN

Tensorflow 2 implementation of the [GatedPixelCNN](https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf).

## Usage

The model can be trained on MNIST or CIFAR10

```shell
python train.py -d mnist
python train.py -d cifar10
```

To see all training options use

```shell
python train.py -h
```