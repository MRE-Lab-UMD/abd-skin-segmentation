# PASCAL VOC and SBD

PASCAL VOC is a standard recognition dataset and benchmark with detection and semantic segmentation challenges.
The semantic segmentation challenge annotates 21 object classes including background.
The Semantic Boundary Dataset (SBD) is a further annotation of the PASCAL VOC data that provides more semantic segmentation and instance segmentation masks.
They can be downloaded from:

- PASCAL VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
- SBD: see [homepage](http://home.bharathh.info/home/sbd) or [direct download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)

This directory contains the splits used to train and test our models. In detail:

- test.txt standard test for PASCAL VOC 2012. (1456 images)
- val.txt PASCAL VOC validation set. (1449 images)
- train.txt is obtained by merging SBD train and validation sets, merging PASCAL VOC train set, discarding duplicates and finally discarding the intersection with PASCAL validation set. (10582 images)
