# PixelSNAIL

Tensorflow 2 implementation of [PixelSNAIL](https://arxiv.org/pdf/1712.09763.pdf.

## Usage

Train on MNIST/CIFAR10 with

```shell
python train.py -d mnist
python train.py -d cifar10
```

See available training options with

```shell
python train.py -h
```

Training data and visualisations can be seen with

```shell
tensorboard --logdir logs
```
