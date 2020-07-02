# PixelCNN

Tensorflow 2 implementation of the original [PixelCNN](https://arxiv.org/pdf/1601.06759.pdf).

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