# Inspired by wkentaro code
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class Metrics(object):
    """ Compute metrics
    This class can compute different metrics. Call only :func: `compute`
    from outside the class because it compute and cache the confusion matrix
    needed for computations
    """
    def __init__(self, n_classes, exclude_background=False, class_weights=[]):
        """
        Args:
            class_weights (list): they are used to compute iiou metric
            usually measured on cityscapes dataset
            exclude_background (bool): whether to include or not background
            class when computing metrics. Default: False
        """
        self.n_classes = n_classes
        self.metrics = {'pixel_acc' : self._pixel_accuracy,
                        'mean_acc' : self._mean_accuracy,
                        'iou_class' : self._iou_class,
                        'iiou_class' : self._iiou_class,
                        'fwiou': self._fwiou}
        self.exclude_background = exclude_background
        self.class_weights = class_weights
        self._reset_cm()

    def _reset_cm(self):
        self.cm = np.zeros((self.n_classes, self.n_classes))

    def _confusion_matrix(self, gt, pred):
        """ Compute confusion matrix
        Used to compute all the metrics.
        rows (first dimension): GT
        cols (second dimension): Prediction
        """
        if self.exclude_background:
            mask = (gt > 0) & (gt < self.n_classes)
        else:
            mask = (gt >= 0) & (gt < self.n_classes)
        return np.bincount(
            self.n_classes * gt[mask].astype(int) +
            pred[mask], minlength=self.n_classes**2).reshape(self.n_classes,
                                                             self.n_classes)

    def compute(self, metric_name, gts, preds):
        """ Compute metrics
        This is the only public function of this class. Compute a single or
        a set of metrics.
        Args:
            metric_name (string or list): the single metric or list to compute
            gts (list of matrices, 2D or 3D matrix): groundtruth
            preds (list of matrices, 2D or 3D matrix): predictions
        """
        self._reset_cm()
        if isinstance(gts,list) or gts.ndim == 3:
            for lt, lp in zip(gts, preds):
                self.cm += self._confusion_matrix(lt.flatten(),
                                                  lp.flatten())
        elif gts.ndim == 2:
            self.cm = self._confusion_matrix(gts.flatten(),
                                              preds.flatten())

        if isinstance(metric_name,list):
            return [self.metrics[m]() for m in metric_name]
        else:
            return self.metrics[metric_name]()

    def _pixel_accuracy(self):
        """Pixel-wise accuracy
        """
        pixel_acc = np.diag(self.cm).sum() / self.cm.sum()
        return pixel_acc

    def _mean_accuracy(self):
        """Pixel-wise accuracy but averaged on classes
        """
        acc_cls = np.diag(self.cm) / self.cm.sum(axis=1)
        return np.nanmean(acc_cls)

    def _iou_class(self):
        """Intersection over Union averaged on classes
            Formula: TP / FP + FN + TP
        Literally from code below:
        TP / (FP+TP) + (FN+TP) - TP
        """
        iou = np.diag(self.cm) / (self.cm.sum(axis=1) +
                                  self.cm.sum(axis=0) -
                                  np.diag(self.cm))
        # If no TP, FP nor FN are present it happens a 0 by 0 division.
        # handle the resulting nans
        return np.nanmean(iou)

    def _iiou_class(self):
        """Intersection over Union averaged on classes weighted by
           average instance size
        """
        tp = np.diag(self.cm) * self.class_weights
        fp = self.cm.sum(axis=1)
        fn = self.cm.sum(axis=0) * self.class_weights
        iiou = tp / (fp + fn - tp)
        return np.nanmean(iiou)

    def _fwiou(self):
        iou = np.diag(self.cm) / (self.cm.sum(axis=1) +
                                  self.cm.sum(axis=0) -
                                  np.diag(self.cm))
        freq = self.cm.sum(axis=1) / self.cm.sum()
        return (freq[freq > 0] * iou[freq > 0]).sum()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class MultiAverageMeter(object):
    """ Wrapper for AverageMeter to handle multiple values at a time
    """
    def __init__(self, n_values):
        self.meters = [AverageMeter() for _ in range(n_values)]

    def update(self, values, n=1):
        if len(values) != len(self.meters):
            raise ValueError('Wrong number of values. Should be {}'
                             ' instead is {}'.format(len(self.meters),
                                                     len(values)))
        for i,v in enumerate(values):
            self.meters[i].update(v,n)



if __name__ == '__main__':
    #This code is for debug purposes
    metrics = Metrics(3)
    pred = np.array([[0,1,0],[1,2,0]])
    gt = np.array([[0,0,0],[1,0,2]])
    values = metrics.compute(['pixel_acc','iou_class'], gt, pred)
    print(list(zip(['pixel_acc','iou_class'],values)))
    values = metrics.compute('iou_class', gt, pred)
    print('iou_class',values)
