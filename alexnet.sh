#!/usr/bin/env bash

echo "Alexnet PRELU3"

echo "ERROR 0.01"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU3 -error 0.01 -net alexnet
echo "ERROR 0.02"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU3 -error 0.02 -net alexnet
echo "ERROR 0.03"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU3 -error 0.03 -net alexnet
echo "ERROR 0.04"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU3 -error 0.04 -net alexnet
echo "ERROR 0.05"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU3 -error 0.05 -net alexnet
echo "ERROR 0.06"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU3 -error 0.06 -net alexnet
echo "ERROR 0.07"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU3 -error 0.07 -net alexnet
echo "ERROR 0.08"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU3 -error 0.08 -net alexnet
echo "ERROR 0.09"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU3 -error 0.09 -net alexnet


echo "Alexnet PRELU4"

echo "ERROR 0.01"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU4 -error 0.01 -net alexnet
echo "ERROR 0.02"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU4 -error 0.02 -net alexnet
echo "ERROR 0.03"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU4 -error 0.03 -net alexnet
echo "ERROR 0.04"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU4 -error 0.04 -net alexnet
echo "ERROR 0.05"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU4 -error 0.05 -net alexnet
echo "ERROR 0.06"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU4 -error 0.06 -net alexnet
echo "ERROR 0.07"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU4 -error 0.07 -net alexnet
echo "ERROR 0.08"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU4 -error 0.08 -net alexnet
echo "ERROR 0.09"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU4 -error 0.09 -net alexnet



echo "Alexnet PRELU5"

echo "ERROR 0.01"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.01 -net alexnet
echo "ERROR 0.02"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.02 -net alexnet
echo "ERROR 0.03"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.03 -net alexnet
echo "ERROR 0.04"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.04 -net alexnet
echo "ERROR 0.05"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.05 -net alexnet
echo "ERROR 0.06"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.06 -net alexnet
echo "ERROR 0.07"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.07 -net alexnet
echo "ERROR 0.08"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.08 -net alexnet
echo "ERROR 0.09"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.09 -net alexnet
