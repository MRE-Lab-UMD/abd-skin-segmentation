import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path

ALPHA = 0.5

def test(args):

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    orig_img = misc.imread(args.img_path)

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True)
    n_classes = loader.n_classes

    img = orig_img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
    img = img.astype(float) / 255.0
    # NHWC -> NCWH
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model = get_model(args.arch, n_classes)
    model.load_state_dict(torch.load(args.model_path)['state_dict'])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
    model.eval()

    if torch.cuda.is_available():
        model.cuda(0)
        images = Variable(img.cuda(0))
    else:
        images = Variable(img)

    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = loader.decode_segmap(pred)
    if args.alpha_blend:
        orig_img = misc.imresize(orig_img, (loader.img_size[0], loader.img_size[1]))
        out_img = ALPHA * orig_img + (1 - ALPHA) * decoded
    else:
        out_img = decoded
    print(np.unique(pred))
    misc.imsave(args.out_path, out_img)
    print("Segmentation Mask Saved at: {}".format(args.out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None,
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None,
                        help='Path of the output segmap')
    parser.add_argument('--alpha_blend', nargs='?', type=bool, default=True,
                        help='Blend input image with predicted mask')
    args = parser.parse_args()
    test(args)
