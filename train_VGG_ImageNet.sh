#!/usr/bin/env bash

echo "Train VGG16"
for act in RELU PRELU1 PRELU2 PRELU3 PRELU4 PRELU5
do
    python3 train_ImageNet.py -gpuid 0 2 3 4 5 -act $act -net vgg16
done
