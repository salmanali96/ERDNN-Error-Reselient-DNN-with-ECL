#!/usr/bin/env bash

echo "VGG PRELU3"

echo "ERROR 0.01"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU3 -error 0.01 -net VGG
echo "ERROR 0.02"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU3 -error 0.02 -net VGG
echo "ERROR 0.03"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU3 -error 0.03 -net VGG
echo "ERROR 0.04"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU3 -error 0.04 -net VGG
echo "ERROR 0.05"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU3 -error 0.05 -net VGG
echo "ERROR 0.06"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU3 -error 0.06 -net VGG
echo "ERROR 0.07"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU3 -error 0.07 -net VGG
echo "ERROR 0.08"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU3 -error 0.08 -net VGG
echo "ERROR 0.09"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU3 -error 0.09 -net VGG

echo "PRELU4"

echo "ERROR 0.01"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU4 -error 0.01 -net VGG
echo "ERROR 0.02"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU4 -error 0.02 -net VGG
echo "ERROR 0.03"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU4 -error 0.03 -net VGG
echo "ERROR 0.04"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU4 -error 0.04 -net VGG
echo "ERROR 0.05"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU4 -error 0.05 -net VGG
echo "ERROR 0.06"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU4 -error 0.06 -net VGG
echo "ERROR 0.07"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU4 -error 0.07 -net VGG
echo "ERROR 0.08"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU4 -error 0.08 -net VGG
echo "ERROR 0.09"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU4 -error 0.09 -net VGG


echo "PRELU5"

echo "ERROR 0.01"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU5 -error 0.01 -net VGG
echo "ERROR 0.02"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU5 -error 0.02 -net VGG
echo "ERROR 0.03"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU5 -error 0.03 -net VGG
echo "ERROR 0.04"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU5 -error 0.04 -net VGG
echo "ERROR 0.05"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU5 -error 0.05 -net VGG
echo "ERROR 0.06"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU5 -error 0.06 -net VGG
echo "ERROR 0.07"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU5 -error 0.07 -net VGG
echo "ERROR 0.08"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU5 -error 0.08 -net VGG
echo "ERROR 0.09"
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU5 -error 0.09 -net VGG