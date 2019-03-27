import os
import os.path
import collections
import json
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils import data


def get_data_path(name):
    js = open('config.json').read()
    data = json.loads(js)
    return data[name]['data_path']


class pascalVOCLoader(data.Dataset):
    def __init__(self, root, split="train_aug", transform=None):
        self.root = root
        self.img_path = os.path.join(root, 'JPEGImages')
        self.lbl_path = os.path.join(root, 'SegmentationClass', 'pre_encoded')
        self.split = split
        self.transform = transform
        self.n_classes = 21
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)

        for split in ["train", "val", "trainval"]:
            file_list = tuple(open(root + '/ImageSets/Segmentation/' + split + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

        if not os.path.isdir(os.path.join(self.root, 'SegmentationClass', 'pre_encoded')):
            self.setup(pre_encode=True)
        else:
            self.setup(pre_encode=False)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = os.path.join(self.img_path, img_name + '.jpg')
        lbl_path = os.path.join(self.lbl_path, img_name + '.png')

        img = self.loader(img_path, True)
        lbl = self.loader(lbl_path, False)

        if self.transform is not None:
            img, lbl = self.transform(img, lbl)

        return img, lbl.squeeze()

    def loader(self, path, rgb=True):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            if rgb:
                return img.convert('RGB')
            else:
                return img.convert('L')

    def get_pascal_labels(self):
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                              [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                              [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
                              [0,192,0], [128,192,0], [0,64,128]])


    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask


    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup(self, pre_encode=False):
        sbd_path = get_data_path('sbd')
        voc_path = get_data_path('pascal')

        target_path = self.root + '/SegmentationClass/pre_encoded/'
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Load SBD train set
        sbd_train_list = tuple(open(sbd_path + 'dataset/train.txt', 'r'))
        sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
        # Load SBD val set
        sbd_val_list = tuple(open(sbd_path + 'dataset/val.txt', 'r'))
        sbd_val_list = [id_.rstrip() for id_ in sbd_val_list]
        # Join everything
        self.files['train_aug'] = self.files['train'] + sbd_train_list + sbd_val_list
        # Remove duplicates and intersection with Pascal VOC validation set
        self.files['train_aug'] = list(set(self.files['train_aug']) - set(self.files['val']))
        self.files['train_aug'].sort()

        if pre_encode:
            print("Pre-encoding segmentation masks...")
            lbl_dir = os.path.join(sbd_path, 'dataset', 'cls')
            lbl_list = [f for f in os.listdir(lbl_dir) if f.endswith('.mat')]
            for i in tqdm(lbl_list):
                lbl_path = os.path.join(lbl_dir, i)
                lbl = io.loadmat(lbl_path)['GTcls'][0]['Segmentation'][0].astype(np.int32)
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(target_path + os.path.splitext(i)[0] + '.png', lbl)

            for i in tqdm(self.files['trainval']):
                lbl_path = self.root + '/SegmentationClass/' + i + '.png'
                lbl = self.encode_segmap(m.imread(lbl_path))
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(target_path + i + '.png', lbl)

if __name__ == '__main__':
    import sys
    sys.path.append(".")
    from ptsemseg.loader import get_loader, get_data_path
    dst = pascalVOCLoader(get_data_path('pascal'), is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            plt.imshow(dst.decode_segmap(labels.numpy()[i+1]))
            plt.show()
