# Conditional Neural Processes

Tensorflow 2 generic implementation of the [conditional neural processes (CNP)](https://arxiv.org/pdf/1807.01613.pdf).

## Usage

To train the model on a regressive task or on MNIST

```shell
python train.py -t mnist
python train.py -r regression
```

To see all training options use

```shell
python train.py -h
```