""" helper function

author baiyu
"""

import os
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
from collections import OrderedDict


def LoadModel(args, model):
    PATH = './checkpoint/ImageNet/' + args.net + '/' + args.net + '_' + args.act + '.pth'
    PATH = os.path.abspath(PATH)
    if args.gpu:
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage.cuda(args.gpuids[0]))
        try:
            model.load_state_dict(checkpoint)
        except:
            model.module.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
        try:
            model.load_state_dict(checkpoint)
        except:
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k[:7] == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k[:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
    return model


def get_network(args, use_gpu=True):
    """ return given network
    """
    if args.net == 'vgg16':
        from models.ImageNet import VGG
        net = VGG.vgg16_bn(args)
    elif args.net == 'resnet':
        from models.ImageNet import resnet
        net = resnet.resnet18(args)
    elif args.net == 'alexnet':
        from models.ImageNet import alexnet
        net = alexnet.alexnet(args)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        torch.cuda.set_device(args.gpuids[0])
        with torch.cuda.device(args.gpuids[0]):
            net = net.cuda()
        net = nn.DataParallel(net, device_ids=args.gpuids, output_device=args.gpuids[0])
        cudnn.benchmark = True

    if args.resume == 'yes':
        net = LoadModel(args, net)

    return net


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super(WarmUpLR,self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
