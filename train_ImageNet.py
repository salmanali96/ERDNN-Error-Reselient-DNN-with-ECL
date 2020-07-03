# train_ImageNet.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from conf import settings
from utils_ImageNet import WarmUpLR, get_network, accuracy
from progressbar import ProgressBar, Percentage, Bar, ETA, SimpleProgress
from math import ceil

# for ignore imagenet PIL EXIF UserWarning
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def train(epoch):
    net.train()
    train_loss = 0.0

    widgets = ['Train: ', Percentage(), ' ',
               Bar(marker='#',left='[',right=']'),
               ' ', ETA(), '  batch: (', SimpleProgress(sep='/'), ')']
    pbar = ProgressBar(widgets=widgets, term_width=80, maxval=ceil(float(len(training_loader.dataset))/float(args.tb)))
    pbar.start()
    for batch_index, (images, labels) in enumerate(training_loader):
        labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = net(images)

        loss = loss_function(outputs, labels)
        train_loss += loss.item()
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        loss.backward()
        optimizer.step()
        if epoch <= args.warm:
            warmup_scheduler.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1
        last_layer = list(net.children())[-1]

        pbar.update(batch_index)
    pbar.finish()

    print('Train set: Average Loss: {:.4f}, Accuracy: {:.4f}({:.4f})'.format(
        train_loss / len(training_loader.dataset),
        acc1, acc5
    ))

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]


def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    widgets = ['Test: ', Percentage(), ' ',
               Bar(marker='#',left='[',right=']'),
               ' ', ETA(), '  batch: (', SimpleProgress(sep='/'), ')']
    pbar = ProgressBar(widgets=widgets, term_width=80, maxval=ceil(float(len(test_loader.dataset))/float(args.vb)))
    pbar.start()
    for batch_index, (images, labels) in enumerate(test_loader):
        labels = labels.cuda(non_blocking=True)
        outputs = net(images)

        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        pbar.update(batch_index)
    pbar.finish()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}({:.4f})'.format(
        test_loss / len(test_loader.dataset),
        acc1, acc5
    ))

    return correct.float() / len(test_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg16', help='net type (default: vgg16)')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpuids', default=[0], nargs='+', help='GPU IDs for using (Default: 0)')
    parser.add_argument('-w', type=int, default=8, help='number of workers for dataloader (default: 8)')
    parser.add_argument('-tb', type=int, default=128, help='training batch size (default: 128)')
    parser.add_argument('-vb', type=int, default=32, help='validation batch size (default: 32)')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase (default: 1)')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate (default: 0.01)')
    parser.add_argument('-wd', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('-momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('-act', type=str, default='PRELU1', help='Activation function to use (default: PRELU1)')
    parser.add_argument('-error', type=float, default=0.01, help='Error Rate (default: 0.01)')
    parser.add_argument('-resume', type=str, default='no', help='Resume the training (default: no)')
    parser.add_argument('-datapath', type=str, default='./data', help='Data path (default: ./data)')

    args = parser.parse_args()
    args.gpuids = list(map(int, args.gpuids))
    print ('[ {}-{} ]'.format(args.net, args.act))
    net = get_network(args, use_gpu=args.gpu)

    # data preprocessing:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = torchvision.datasets.ImageNet(
        root=args.datapath, split='train', download=False,
        transform=transform_train)
    testset = torchvision.datasets.ImageNet(
        root=args.datapath, split='val', download=False,
        transform=transform_test)
    training_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.tb, shuffle=True,
            num_workers=args.w, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.vb, shuffle=False,
            num_workers=args.w, pin_memory=True)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES_IMAGENET, gamma=0.1) # learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'ImageNet', args.net)
    path = './checkpoint/ImageNet/' + args.net + '/' + args.net + '_' + args.act + '.pth'

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    best_acc = 0.0
    for epoch in range(0, settings.EPOCH_IMAGENET):
        print('\nEpoch: {}, Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        train(epoch)
        acc = eval_training(epoch)
        if epoch > args.warm:
            train_scheduler.step()

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES_IMAGENET[0] and best_acc < acc:
            torch.save(net.state_dict(), path)
            best_acc = acc
            continue
