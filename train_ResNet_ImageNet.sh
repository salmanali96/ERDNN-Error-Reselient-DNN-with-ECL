#!/usr/bin/env bash

echo "Train ResNet18"
# for act in RELU PRELU1 PRELU2 PRELU3 PRELU4 PRELU5
for act in RELU PRELU2 PRELU5
do
    python3 train_ImageNet.py -gpuid 0 2 3 4 5 -act $act -net resnet -tb 256 -vb 256
done
