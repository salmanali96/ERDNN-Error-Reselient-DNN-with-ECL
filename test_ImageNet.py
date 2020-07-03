#test_ImageNet.py
#!/usr/bin/env python3


""" test neuron network performace
print top1 and top5 err on test dataset
of a model
"""

import csv
import shutil
import argparse
from os import remove
from os.path import isfile

import torch
import torchvision
import torchvision.transforms as transforms

from conf import settings
from utils_ImageNet import get_network
from progressbar import ProgressBar, Percentage, Bar, ETA, SimpleProgress
from math import ceil

# for ignore imagenet PIL EXIF UserWarning
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg16', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpuids', default=[0], nargs='+', help='GPU IDs for using (Default: 0)')
    parser.add_argument('-w', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-act', type=str, default='PRELU1', help='Activation function to use')
    parser.add_argument('-error', type=float, default=0.01, help='Error Rate')
    parser.add_argument('-resume', type=str, default='yes', help='resume from checkpoint')
    parser.add_argument('-datapath', type=str, default='./data', help='data path')

    args = parser.parse_args()
    args.gpuids = list(map(int, args.gpuids))

    net = get_network(args, use_gpu=args.gpu)

    # data preprocessing:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    testset = torchvision.datasets.ImageNet(
        root=args.datapath, split='val', download=False,
        transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.b, shuffle=False,
            num_workers=args.w, pin_memory=True)
    
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    widgets = ['Test: ', Percentage(), ' ',
               Bar(marker='#',left='[',right=']'),
               ' ', ETA(), '  batch: (', SimpleProgress(sep='/'), ')']
    pbar = ProgressBar(widgets=widgets, term_width=80, maxval=ceil(float(len(test_loader.dataset))/float(args.b)))
    pbar.start()
    for n_iter, (images, labels) in enumerate(test_loader):
        output = net(images)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        labels = labels.cuda(non_blocking=True)
        labels = labels.view(labels.size(0), -1).expand_as(pred)
        correct = pred.eq(labels).float()

        #compute top5
        correct_5 += correct[:, :5].sum()

        #compute top1
        correct_1 += correct[:, :1].sum()
        pbar.update(n_iter)
    pbar.finish()

    top1 = float(str(100 * correct_1.float() / len(test_loader.dataset))[7:-18])
    top5 = float(str(100 * correct_5.float() / len(test_loader.dataset))[7:-18])
    num_params = sum(p.numel() for p in net.parameters())

    print("Top 1 acc: {:.4f}".format(top1))
    print("Top 5 acc: {:.4f}".format(top5))
    print("Parameter numbers: {}".format(num_params))

    # summary_file = 'test_results_of_models_on_error_without_ECL.csv'
    summary_file = 'test_results_of_models_on_error_with_ECL.csv'
    # summary_file = 'test_results_of_models_without_error.csv'
    if not isfile(summary_file):
        with open(summary_file, 'w', newline='') as csv_out:
            writer = csv.writer(csv_out)
            header_list = ['Bit Error Rate', 'Model', 'Activation', 'Top-1 acc', 'Top-5 acc', '#Params']
            # header_list = ['Model', 'Activation', 'Top-1 acc', 'Top-5 acc', '#Params']
            writer.writerow(header_list)
            writer.writerow([args.error, args.net, args.act, top1, top5, num_params])
            # writer.writerow([args.net, args.act, top1, top5, num_params])
    else:
        file_temp = 'temp.csv'
        shutil.copyfile(summary_file, file_temp)
        with open(file_temp, 'r', newline='') as csv_in:
            with open(summary_file, 'w', newline='') as csv_out:
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)
                for row_list in reader:
                    writer.writerow(row_list)
                writer.writerow([args.error, args.net, args.act, top1, top5, num_params])
                # writer.writerow([args.net, args.act, top1, top5, num_params])
        remove(file_temp)
