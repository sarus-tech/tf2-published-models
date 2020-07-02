# PixelCNN++

Tensorflow 2 implementation of OpenAI's [PixelCNN++](https://arxiv.org/pdf/1701.05517.pdf) improvement over gated PixelCNN.

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