# Error Correction Layer with PwReLU

## Requirements

- python 3.5+
- pytorch 1.0+
- torchvision 0.4+
- tqdm (for progress bar)
- numpy

## Project Guidelines
- Checkpoint folder contains the google drive link for all the pre-trained models.
- After downloading the trained model file, copy it into its respective folder
- You can create custom look-up-table from the LUT.py file. Errors are injected into the model using these Look-up-Tables


## Pretrained model setting for ImageNet

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

## Test for MNIST

By running the train_MNIST.py file, you can test the performance on MNIST dataset

python train_MNIST.py -net lenet -act PRELU3 -resume yes -error 0.1


## Test for CIFAR10 

By running the train_CIFAR10.py file, you can test the performance on CIFAR10 dataset

python3 train_CIFAR10.py -act PRELU5 -error 0.02 -net VGG

## Test for CIFAR100

By running the test.py file, you can test the performance on CIFAR100 dataset

python3 train_CIFAR10.py -act PRELU5 -error 0.02 -net VGG

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
