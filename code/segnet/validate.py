import sys
import torch
import visdom
import argparse
import time
import numpy as np
import math
import os
import os.path

import scipy.misc as misc
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.loss import cross_entropy2d
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import AverageMeter
from ptsemseg.metrics import MultiAverageMeter
from ptsemseg.metrics import Metrics
from ptsemseg.loader import transforms

ALPHA = 0.5

def validate(valloader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    eval_time = AverageMeter()
    losses = AverageMeter()
    multimeter = MultiAverageMeter(len(args.metrics))
    metrics = Metrics(n_classes=args.n_classes,
                      exclude_background=args.exclude_background)
    model.eval()
    if torch.cuda.is_available() and not isinstance(model, nn.DataParallel):
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    gts, preds = [], []
    end = time.perf_counter()
    for i, (images, labels) in enumerate(valloader):
        if args.max_iters_per_epoch != 0:
            if i > args.max_iters_per_epoch:
                break
        # measure data loading time
        data_time.update(time.perf_counter() - end)
        if torch.cuda.is_available():
            images = Variable(images.cuda(), volatile=True)
            labels = Variable(labels.cuda(async=True), volatile=True)
        else:
            images = Variable(images, volatile=True)
            labels = Variable(labels, volatile=True)

        outputs = model(images)
        start_eval_time = time.perf_counter()
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        values = metrics.compute(args.metrics, gt, pred)
        multimeter.update(values, images.size(0))
        eval_time.update(time.perf_counter() - start_eval_time)

        for ii, (gt_, pred_) in enumerate(zip(gt, pred)):
            gts.append(gt_)
            preds.append(pred_)
            # Save Segmentation Masks
            if len(args.segmentation_maps_path) > 0:
                decoded = val_dataset.decode_segmap(pred_)
                img_idx = i*images.size(0)+ii
                img_name = valloader.dataset.files[args.split][img_idx]
                if args.alpha_blend:
                    orig_img = misc.imread(os.path.join(valloader.dataset.img_path,
                                            img_name + '.jpg'))
                    orig_img = misc.imresize(orig_img, (256,
                                                        256))
                    out_img = ALPHA * orig_img + (1 - ALPHA) * decoded
                else:
                    out_img = decoded
                misc.imsave(os.path.join(args.segmentation_maps_path, img_name + '.png'),
                            out_img)

        loss = criterion(outputs, labels)
        losses.update(loss.data[0], images.size(0))

        #measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        batch_log_str = ('Val: [{}/{}][{}/{}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Eval {eval_time.val:.3f} ({eval_time.avg:.3f})\t'
                        'Loss: {loss.val:.3f} ({loss.avg:.3f})'.format(
                           epoch+1, args.n_epoch, i,
                           math.floor(valloader.dataset.__len__()/valloader.batch_size),
                           batch_time=batch_time, data_time=data_time,
                           eval_time=eval_time, loss=losses))
        for i,m in enumerate(args.metrics):
            batch_log_str += ' {}: {:.3f} ({:.3f})'.format(m ,
                                                           multimeter.meters[i].val,
                                                           multimeter.meters[i].avg)
        print(batch_log_str)

    globalValues = metrics.compute(args.metrics, gts, preds)
    print('Global Metrics:')
    for m,v in zip(args.metrics, globalValues):
        print('{}: {}'.format(m, v))

    return dict(loss = losses, metrics = multimeter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--backend', nargs='?', type=str, default='resnet18',
                        help='Backend to use (available only for pspnet)'
                        'available: squeezenet, densenet, resnet18,34,50,101,152')
    parser.add_argument('--auxiliary_loss', action='store_true',
                        help='Activate auxiliary loss for deeply supervised models')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='val',
                        help='Split of dataset to test on')
    parser.add_argument('--metrics', nargs='?', type=str, default='pixel_acc,iou_class',
                        help='Metrics to compute and show')
    parser.add_argument('--num_workers', nargs='?', type=int, default=4,
                        help='Number of processes to load and preprocess images')
    parser.add_argument('--max_iters_per_epoch', nargs='?', type=int, default=0,
                        help='Max number of iterations per epoch.'
                             ' Useful for debug purposes')
    parser.add_argument('--exclude_background', action='store_true',
                        help='Exclude background class when evaluating')
    parser.add_argument('--segmentation_maps_path', nargs='?', type=str,
                        default='', help='Directory to save segmentation maps'
                        ' leave it blank to disable saving')
    parser.add_argument('--alpha_blend', action='store_true',
                        help='Blend input image with predicted mask')
    args = parser.parse_args()
    #Params preprocessing
    args.metrics = args.metrics.split(',')
    args.n_epoch = 1
    # For now settings for each backend are hardcoded
    args.pspnet_sizes = (1,2,3,6)
    if args.backend == 'squeezenet':
        args.psp_size = 512
        args.deep_features_size = 256
    elif args.backend == 'densenet':
        args.psp_size = 1024
        args.deep_features_size = 512
    elif args.backend == 'resnet18' or args.backend == 'resnet34':
        args.psp_size = 512
        args.deep_features_size = 256
    elif args.backend == 'resnet50' or args.backend == 'resnet101' or args.backend == 'resnet152':
        args.psp_size = 2048
        args.deep_features_size = 1024

    # Setup Dataloader
    val_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    val_dataset = data_loader(data_path,
                      split='val',
                      transform=val_transforms)

    args.n_classes = val_dataset.n_classes
    valloader = data.DataLoader(val_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             pin_memory=True)

    # Setup Model
    model = get_model(args)
    model.load_state_dict(torch.load(args.model_path)['state_dict'])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
    cudnn.benchmark = True

    if len(args.segmentation_maps_path) > 0:
        if not os.path.exists(args.segmentation_maps_path):
            os.makedirs(args.segmentation_maps_path)

    validate(valloader, model, cross_entropy2d, 0, args)
