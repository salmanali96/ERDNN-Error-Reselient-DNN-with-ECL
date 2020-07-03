# Error Correction Layer with PwReLU

## Requirements

- python 3.5+
- pytorch 1.0+
- torchvision 0.4+
- tqdm (for progress bar)
- numpy

## NOW

- training alexnet with ReLU, PwReLU2, PwReLU5 on imagenet at mlvc07 server.
- coding resnet for imagenet

## Pretrained model setting

- vgg16
    - training batch size: 128
- alexnet
    - training batch size: 256
- resnet18
    - training batch size: 256

## Make LookUp Table for Error Correction Layer

### Usage

```shell
$ ./Error_Files/make_LUT.sh
```

or

```shell
$ python3 ./Error_Files/LUT.py -e 0.01
```

## Test for ImageNet

### Usage

**for VGG16 trained on ImageNet on Single GPU with PwReLU1 and 0.01 bit error rate**

```shell
$ python3 test_ImageNet.py -act PRELU1 -error 0.01
```

## Reference

- [pytorch official imagenet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py)
- [progressbar](https://github.com/niltonvolpato/python-progressbar)
# ERDNN-Error-Reselient-DNN-with-ECL
