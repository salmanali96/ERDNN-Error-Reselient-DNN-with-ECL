#!/usr/bin/env bash

for arch in vgg16 alexnet resnet
do
    # for act in RELU PRELU1 PRELU2 PRELU3 PRELU4 PRELU5
    for act in RELU PRELU2 PRELU5
    do
        # for err in 0.02 0.04 0.06
        for err in 0.001 0.005 0.01
        do
            if [ "$arch" = "vgg16" ]; then
                echo "    [ $arch-$act test on error rates of $err ]"
                python3 test_ImageNet.py -gpuids 0 2 3 4 5 -b 32 -net $arch -act $act -error $err
            else
                echo "    [ $arch-$act test on error rates of $err ]"
                python3 test_ImageNet.py -gpuids 0 2 3 4 5 -b 128 -net $arch -act $act -error $err
            fi
        done
    done
done
