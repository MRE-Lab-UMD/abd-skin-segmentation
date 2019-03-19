import sys
import os.path
import torch
import visdom
import argparse
import random
import time
import math

from PIL import Image
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data
from torch.optim.lr_scheduler import MultiStepLR

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.metrics import AverageMeter
from ptsemseg.metrics import MultiAverageMeter
from ptsemseg.metrics import Metrics
from ptsemseg.utils import save_checkpoint
from ptsemseg.loader import transforms
from validate import validate

best_metric_value = 0
# Setup visdom for visualization
vis = visdom.Visdom()

def main(args):
    global best_metric_value

    train_transforms = transforms.Compose([transforms.Resize(512, minside=False),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(10, resample=Image.BILINEAR),
                                           transforms.PadToSize(480),
                                           transforms.RandomResizedCrop(480, scale=(0.5, 2), ratio=(1, 1)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([transforms.Resize(512, minside=False),
                                         transforms.PadToSize(480),
                                         transforms.CenterCrop(480),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    # Setup Dataset and Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    # Train
    train_dataset = data_loader(data_path,transform=train_transforms)
    args.n_classes = train_dataset.n_classes
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  pin_memory=True)
    # Validation
    val_dataset = data_loader(data_path,
                      split='val',
                      transform=val_transforms)
    valloader = data.DataLoader(val_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             pin_memory=True)

    # Setup Model
    model = get_model(args)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_metric_value = checkpoint['best_metric_value']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
        cudnn.benchmark = True

    if args.lr_policy == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in args.milestones.split(',')])

    loss_viswindow = vis.line(X=torch.zeros((1, )).cpu(),
                              Y=torch.zeros((1, 2)).cpu(),
                              opts=dict(xlabel='Epochs',
                                        ylabel='Loss',
                                        title='Loss trough Epochs',
                                        legend=['Train','Val']))

    # Open log file
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    log_file = open(os.path.join(args.save_path, 'logs.txt'), 'w')
    log_header = 'epoch'
    log_header += ',train_loss'
    for m in args.metrics:
        log_header += ',train_' + m
    log_header += ',val_loss'
    for m in args.metrics:
        log_header += ',val_' + m
    log_file.write(log_header + '\n')

    # Main training loop
    for epoch in range(args.start_epoch, args.n_epoch):

        trainmetrics = train(trainloader, model, cross_entropy2d, optimizer, epoch, args)
        args.split='val'
        valmetrics = validate(valloader, model, cross_entropy2d, epoch, args)
        if args.lr_policy == "MultiStepLR":
            scheduler.step()

        # Write log file
        log_line = '{}'.format(epoch)
        log_line += ',{:.3f}'.format(trainmetrics['loss'].avg)
        for m in trainmetrics['metrics'].meters:
            log_line += ',{:.3f}'.format(m.avg)
        log_line += ',{:.3f}'.format(valmetrics['loss'].avg)
        for m in valmetrics['metrics'].meters:
            log_line += ',{:.3f}'.format(m.avg)
        log_file.write(log_line + '\n')

        # Track loss trough epochs
        vis.line(
            X=torch.ones((1,2)).cpu()*epoch,
            Y=torch.Tensor([trainmetrics['loss'].avg, valmetrics['loss'].avg]).unsqueeze(0).cpu(),
            win=loss_viswindow,
            update='append')

        # Take best and save model
        curr_metric_value = valmetrics['metrics'].meters[0].avg
        is_best = curr_metric_value > best_metric_value
        best_metric_value = max(curr_metric_value, best_metric_value)
        if epoch % args.save_every == 0 and epoch != 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'args': args,
                'state_dict': model.module.state_dict(),
                'best_metric_value': best_metric_value,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_path,
                            "{}_{}_{}.pth".format(args.arch,
                                                  args.dataset,
                                                  epoch)))
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'args': args,
                'state_dict': model.module.state_dict(),
                'best_metric_value': best_metric_value,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_path, 'model_best.pth.tar'))

    log_file.close()

def train(trainloader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    eval_time = AverageMeter()
    losses = AverageMeter()
    multimeter = MultiAverageMeter(len(args.metrics))
    metrics = Metrics(n_classes=args.n_classes,
                      exclude_background=args.exclude_background)

    # Initialize current epoch log
    if epoch==0:
        epoch_loss_window = vis.line(X=torch.zeros(1),
                               Y=torch.zeros(1),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Epoch {} Training Loss'.format(epoch),
                                         legend=['Loss']))

    model.train()

    end = time.perf_counter()
    for i, (images, labels) in enumerate(trainloader):
        if args.max_iters_per_epoch != 0:
            if i > args.max_iters_per_epoch:
                break
        # measure data loading time
        data_time.update(time.perf_counter() - end)
        if torch.cuda.is_available():
            images = Variable(images.cuda(0, async=True))
            labels = Variable(labels.cuda(0, async=True))
        else:
            images = Variable(images)
            labels = Variable(labels)

        batch_size = images.size(0)

        # Forward pass
        outputs = model(images)

        # Compute metrics
        start_eval_time = time.perf_counter()
        # sample to lighten evaluation
        sample_idx = random.randint(0,batch_size-1)
        pred = outputs.data[sample_idx,:,:,:].max(0)[1].cpu().numpy()
        gt = labels.data[sample_idx,:,:].cpu().numpy()
        values = metrics.compute(args.metrics, gt, pred)
        multimeter.update(values, batch_size)
        eval_time.update(time.perf_counter() - start_eval_time)

        loss = criterion(outputs, labels)
        losses.update(loss.data[0], batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch==0:
            vis.line(
                X=torch.ones(1) * i,
                Y=torch.Tensor([loss.data[0]]),
                win=epoch_loss_window,
                update='append')

        batch_log_str = ('Epoch: [{}/{}][{}/{}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Eval {eval_time.val:.3f} ({eval_time.avg:.3f})\t'
                        'Loss: {loss.val:.3f} ({loss.avg:.3f})'.format(
                           epoch+1, args.n_epoch, i,
                           math.floor(trainloader.dataset.__len__()/trainloader.batch_size),
                           batch_time=batch_time, data_time=data_time,
                           eval_time=eval_time, loss=losses))
        for mi,mn in enumerate(args.metrics):
            batch_log_str += ' {}: {:.3f} ({:.3f})'.format(mn ,
                                                           multimeter.meters[mi].val,
                                                           multimeter.meters[mi].avg)
        print(batch_log_str)

        #measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

    return dict(loss = losses, metrics = multimeter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')

    # Architecture -------------------------------------------------------------
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--backend', nargs='?', type=str, default='resnet18',
                        help='Backend to use (available only for pspnet)'
                        'available: squeezenet, densenet, resnet18,34,50,101,152')
    parser.add_argument('--auxiliary_loss', action='store_true',
                        help='Activate auxiliary loss for deeply supervised models')

    # Data ---------------------------------------------------------------------
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help='Height of the input image')

    # Learning hyperparams -----------------------------------------------------
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--lr_policy', nargs='?', type=str, default='MultiStepLR',
                        help='Adopted learning rate policy: MultiStepLR or PolyLR')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=5e-4,
                        help='Weight Decay')
    parser.add_argument('--milestones', nargs='?', type=str, default='10,20,30',
                        help='Milestones for LR decreasing when using MultiStepLR')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')

    # Others -------------------------------------------------------------------
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_path', nargs='?', type=str, default='.',
                        help='Location where checkpoints are saved')
    parser.add_argument('--save_every', nargs='?', type=int, default=10,
                        help='Save model every x epochs.')
    parser.add_argument('--metrics', nargs='?', type=str, default='pixel_acc,iou_class',
                        help='Metrics to compute and show, the first in the list '
                             'is also used to evaluate the best model to save')
    parser.add_argument('--num_workers', nargs='?', type=int, default=4,
                        help='Number of processes to load and preprocess images')
    parser.add_argument('--max_iters_per_epoch', nargs='?', type=int, default=0,
                        help='Max number of iterations per epoch.'
                             ' Useful for debug purposes')
    parser.add_argument('--exclude_background', action='store_true',
                        help='Exclude background class when evaluating')
    parser.add_argument('--segmentation_maps_path', nargs='?', type=str,
                        default='', help='Directory to save segmentation maps'
                        'when validating. Leave it blank to disable saving')
    parser.add_argument('--alpha_blend', action='store_true',
                        help='Blend input image with predicted mask when saving'
                        ' (only in validation)')

    args = parser.parse_args()
    #Params preprocessing
    args.metrics = args.metrics.split(',')

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

    # Call main function
    main(args)
