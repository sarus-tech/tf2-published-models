# Gated PixelCNN

Tensorflow 2 implementation of the [GatedPixelCNN](https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf).

## Usage

The model can be trained on MNIST or CIFAR10

```shell
python train.py -d mnist
python train.py -d cifar10
```

The model can be trained unconditionally or trained conditionally on class indexes one-hot representations. To train conditionnaly use the `-c` option.

```shell
python train.py -c
```

To see all training options use

```shell
python train.py -h
```