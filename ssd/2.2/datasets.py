import torch
from torch.utils.data import Dataset
import json
import os
from utils import transform

from PIL import Image,ImageDraw
Image.MAX_IMAGE_PIXELS = None
import copy
import matplotlib.pyplot as plt
import numpy as np
def show(imgT):
    plt.imshow(imgT.permute(1,2,0))
    plt.show()

class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, input_size, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.input_size  = input_size
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')        
        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        #print(objects)
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        #print(boxes)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        #print(labels)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)
        #============draw pic for exam===========
        '''image_o=copy.deepcopy(image)
        coors=boxes
        for i in range(len(coors)):
            coors_set1=[(coors[i][0],coors[i][1]),(coors[i][2],coors[i][1]),(coors[i][2],coors[i][3]),(coors[i][0],coors[i][3]),(coors[i][0],coors[i][1])]
            draw = ImageDraw.Draw(image_o)
            draw.line(coors_set1,width=10,fill='red')
        image_o.show()'''
        #============draw pic for exam===========
        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, self.input_size, difficulties, split=self.split)        
        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
