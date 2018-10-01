import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import os.path as osp

from PIL import Image

import torch
from torch.utils.data import Dataset 

import numpy as np
import pandas as pd
import os.path as osp

from PIL import Image

# from encoder import DataEncoder

# from transforms import AffineAugmentation, EncoderTransform
import math

from torchvision import transforms


def load_datalist(path):
    ''' Load COCO GT
    
    Adapted from
    https://github.com/VisionLearningGroup/visda-2018-public/blob/master/detection/convert_datalist_gt_to_pkl.py
    '''
    
    coco_boxes = []
    coco_labels = []
    fnames = []
    for line in open(path, 'r').readlines():
        parts = [p.strip() for p in line.split()]
        fnames.append(parts[0])

        parts = parts[1:]
        boxes = []
        labs = []
        for i in range(0, len(parts), 5):
            x0, y0, x1, y1, cls = parts[i:i+5]
            x0 = float(x0)
            y0 = float(y0)
            x1 = float(x1)
            y1 = float(y1)
            cls = int(cls)
            boxes.append([x0, y0, x1, y1])
            labs.append(cls)
        coco_boxes.append(np.array(boxes))
        coco_labels.append(np.array(labs))

    return fnames, coco_boxes, coco_labels

class VisdaDetectionLoader(Dataset):
    
    id2label  = {
        0 : "aeroplane",
        1 : "bicycle",
        2 : "bus",
        3 : "car",
        4 : "horse",
        5 : "knife",
        6 : "motorcycle",
        7 : "person",
        8 : "plant",
        9 : "skateboard",
        10 : "train",
        11 : "truck"
    }
    
    def __init__(self, root, labels, transforms = None, joint_transforms = None):

        """
        root :  
            directory with image files. labels found in the label file will be combined with this
            directory using `os.path.join`
        labels:
            filename to csv file containing image filenames, bounding boxes and labels in the MS COCO
            format
        transforms:
            transforms to be applied to images right after loading. Last transform should involve
            casting the image to a pytorch tensor
        joint_tranforms:
            transforms jointly applied to the tuple of (image, bounding boxes, labels)
        """
        
        super().__init__()
        
        self.labelfile        = labels
        self.imgroot          = root
        self.transforms       = transforms
        self.joint_transforms = joint_transforms

        self._load_labels()
        
    def __len__(self):

        return len(self.samples)
    
    def __getitem__(self, index):
        
        sample  = self.samples.loc[index]

        im = Image.open(osp.join(self.imgroot, sample.fname))
        bboxes = torch.from_numpy(sample.boxes).float()
        labels = torch.from_numpy(sample.labels).float()
 
        if self.transforms is not None:
            im = self.transforms(im)

        if self.joint_transforms is not None:
            im, bboxes, labels = self.joint_transforms( [im, bboxes, labels] )

        return im, bboxes, labels

    def _load_labels(self):
        
        fnames, coco_boxes, coco_labels = load_datalist(self.labelfile)
        self.samples = pd.DataFrame(data = {'fname' : fnames,
                    'boxes' : coco_boxes, 'labels' : coco_labels})

class MaxTransform():

    def __call__(self, args):

        x,bboxes,y = args

        bboxes_ = bboxes.clone()

        bboxes[:,0] = torch.min(bboxes_[:,0], bboxes_[:,2])
        bboxes[:,1] = torch.min(bboxes_[:,1], bboxes_[:,3])
        bboxes[:,2] = torch.max(bboxes_[:,0], bboxes_[:,2])
        bboxes[:,3] = torch.max(bboxes_[:,1], bboxes_[:,3])

        return (x, bboxes, y)


class CoordShuffle():

    def __call__(self, args):

        x,bboxes,y = args

        bboxes_ = bboxes.clone()
        bboxes[:,0] = bboxes_[:,1]
        bboxes[:,1] = bboxes_[:,0]
        bboxes[:,2] = bboxes_[:,3]
        bboxes[:,3] = bboxes_[:,2]

        return (x, bboxes, y)

def build_dataset(batch_size, which='train_visda',
                  num_workers = None, encode = True,
                  augment=True, shuffle = True):
    import pandas as pd

    data = {
    'train_visda' : {
        'labels' : '/gpfs01/bethge/home/sschneider/data/visda-detection/visda18-detection-train.txt',
        'root'   : '/tmp/visda-detect/png_json/'
    },
    'val_coco' : {
        'labels' : '/gpfs01/bethge/home/sschneider/data/visda-detection/coco17-val.txt',
        'root'   : '/tmp/visda-detect/val2017/'
    },
    'test_visda' : {
        'labels' : '/gpfs01/bethge/home/sschneider/data/visda-detection/visda18-detection-test.txt',
        'root'   : '/tmp/visda-detect/png_json/'
    }
    }


    datasets = pd.DataFrame(data)

    T = transforms.Compose([
        transforms.ColorJitter(.1, .8, .75, 0),
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225] ) 
    ]
    )        

    Tj = transforms.Compose([
        AffineAugmentation(
            coord_order = 'xy',
            flip_x  = 0.5,
            flip_y  = 0.,
            shear_x = (0,0.01),
            shear_y = (0,0.01),
            scale   = (1., 5.),
            rotate  = (-math.pi/20, math.pi/20),
            dx      = (-1,1),
            dy      = (-1,1)
        ) if augment else lambda args: args,
        CoordShuffle(),
        MaxTransform(),
        EncoderTransform() if encode else lambda args: args
    ])

    dataset = VisdaDetectionLoader(**datasets[which].to_dict(),
                                transforms=T, joint_transforms = Tj)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    return loader
        
def build_validation(batch_size, which='train_visda',
                  num_workers = None, encode = True,
                  augment=True, shuffle = True):
    import pandas as pd

    data = {
    'train_visda' : {
        'labels' : '/gpfs01/bethge/home/sschneider/data/visda-detection/visda18-detection-train.txt',
        'root'   : '/tmp/visda-detect/png_json/'
    },
    'val_coco' : {
        'labels' : '/gpfs01/bethge/home/sschneider/data/visda-detection/coco17-val.txt',
        'root'   : '/tmp/visda-detect/val2017/'
    },
    'test_visda' : {
        'labels' : '/gpfs01/bethge/home/sschneider/data/visda-detection/visda18-detection-test.txt',
        'root'   : '/tmp/visda-detect/png_json/'
    }
    }


    datasets = pd.DataFrame(data)

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225] ) 
    ]
    )        

    Tj = transforms.Compose([
        lambda args: args[0], torch.zeros(len(args[0])), torch.zeros(len(args[0]))
    ])

    dataset = VisdaDetectionLoader(**datasets[which].to_dict(),
                                transforms=T, joint_transforms = Tj)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    return loader
        