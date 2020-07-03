#!/usr/bin/env bash

echo "    [ Train vgg16-PwReLU4 ]"
python3 train_ImageNet.py -gpuid 0 2 3 4 5 -net vgg16 -act PRELU4 -tb 128 -vb 32

# for arch in vgg16 alexnet resnet
for arch in alexnet resnet
do
    # for act in RELU PRELU1 PRELU2 PRELU3 PRELU4 PRELU5
    for act in PRELU1 PRELU3 PRELU4
    do
        if [ "$arch" = "vgg16" ]; then
            echo "    [ Train $arch-$act ]"
            python3 train_ImageNet.py -gpuids 0 2 3 4 5 -net $arch -act $act -tb 128 -vb 32
        else
            echo "    [ Train $arch-$act ]"
            python3 train_ImageNet.py -gpuids 0 2 3 4 5 -net $arch -act $act -tb 256 -vb 256
        fi
    done
done
