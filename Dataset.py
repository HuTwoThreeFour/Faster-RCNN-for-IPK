import os
import glob
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET

#'''choose which kind of objects u want'''
#background = ['blank', 'with_coin', 'with_paper']



#choose_dataset = ['Fast-RCNN', 'YOLO']


class Schraubenerkennung(torch.utils.data.Dataset):


# Args:
#      root(string): Root directory
#      background: different background, default: 'blank', man can choose: 'blank', 'with_coin' or 'with_paper'
#      which_object: Schraube or Mutter, default: 'all', man can choose: 'all', 'Schraube' or 'Mutter'


    def __init__(self, root, background='blank', which_object = 'all', transforms= None):
        self.root = root
        self.background = background
        self.which_object = which_object
        self.transforms = transforms
        self.imgs = glob.glob(os.path.join(self.root, self.background, self.which_object,'*.jpg'))   #when pictures in other format, pls change it
        #self.annotations = glob.glob(os.path.join(self.root, self.background, self.which_object, 'PASCAL_VOC', '*.xml'))

    def __getitem__(self, idx):
        self.imgs.sort()
        annotations = glob.glob(os.path.join(self.root, self.background, self.which_object,'PASCAL_VOC', '*.xml'))
        annotations.sort()
        img_path = self.imgs[idx]
        anotation_path = annotations[idx]
        #print(img_path)
        #print(anotation_path)
        img = Image.open(img_path).convert('RGB')
        anotation = ET.parse(anotation_path)
        obj = anotation.find('object')
        boxes = []
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        #print('origin bbox is: ',boxes)
        #print('the dimension of boxes is: ', boxes.dim())
        #print('the size of bbox is: ',boxes.size())

        get_classes = obj.find('name').text  # man should in next steps change the format of classes
        # 00_Mutter_2.5 , 01_Shraube_2.5_3 , 02_Mutter_3 , 03_Shraube_3_12 , 04_Mutter_4 , 05_Schraube_4_14 ,
        # 06_Schraube_5_16 , 07_Mutter_6 , 08_Schraube_6_20 , 09_Mutter_8 , 10_Schraube_8_16 , 11_Schraube_8_30
        Classes = ['Mutter_2.5', 'Shraube_2.5_3', 'Mutter_3', 'Shraube_3_12', 'Mutter_4', 'Schraube_4_14',
                   'Schraube_5_16', 'Mutter_6', 'Schraube_6_20', 'Mutter_8', 'Schraube_8_16', 'Schraube_8_30']
        class_id = []
        class_id.append(Classes.index(get_classes))
        class_id = torch.as_tensor(class_id, dtype=torch.int64)
        #print('class id is: ', class_id)
        #print('the dimension of class is: ', class_id.dim())
        #print('the size of class is: ', class_id.size())

        image_id = torch.tensor([idx], dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(1, dtype=torch.uint8)
        #print(iscrowd)

        target = {}
        target['boxes'] = boxes
        target['labels'] = class_id
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            #img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


# def _load_pascal_annotation(self, index):
#     """
# Load image and bounding boxes info from XML file in the PASCAL VOC
# format.
# """
#     filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
#     tree = ET.parse(filename)
#     objs = tree.findall('object')
#     if not self.config['use_diff']:
#         # Exclude the samples labeled as difficult
#         non_diff_objs = [
#             obj for obj in objs if int(obj.find('difficult').text) == 0
#         ]
#         # if len(non_diff_objs) != len(objs):
#         #     print 'Removed {} difficult objects'.format(
#         #         len(objs) - len(non_diff_objs))
#         objs = non_diff_objs
#     num_objs = len(objs)
#
#     boxes = np.zeros((num_objs, 4), dtype=np.uint16)
#     gt_classes = np.zeros((num_objs), dtype=np.int32)
#     overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
#     # "Seg" area for pascal is just the box area
#     seg_areas = np.zeros((num_objs), dtype=np.float32)
#
#     # Load object bounding boxes into a data frame.
#     for ix, obj in enumerate(objs):
#         bbox = obj.find('bndbox')
#         # Make pixel indexes 0-based
#         x1 = float(bbox.find('xmin').text) - 1
#         y1 = float(bbox.find('ymin').text) - 1
#         x2 = float(bbox.find('xmax').text) - 1
#         y2 = float(bbox.find('ymax').text) - 1
#         cls = self._class_to_ind[obj.find('name').text.lower().strip()]
#         boxes[ix, :] = [x1, y1, x2, y2]
#         gt_classes[ix] = cls
#         overlaps[ix, cls] = 1.0
#         seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
#
#     overlaps = scipy.sparse.csr_matrix(overlaps)
#
#     return {
#         'boxes': boxes,
#         'gt_classes': gt_classes,
#         'gt_overlaps': overlaps,
#         'flipped': False,
#         'seg_areas': seg_areas
#     }
