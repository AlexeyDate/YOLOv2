import os
import xmltodict
import torch
import numpy as np
import albumentations as alb
import albumentations.pytorch
from PIL import Image
from torch.utils.data import Dataset
from tools.utils import inersection_over_union_anchors


class Dataset(Dataset):
    """
    Base class of txt and xml description files.
    Class can convert [x1, y1, x2, y2] view to the YOLO standart format
    and create a target matrix after creating dataloader.

    Note: shape of target = (batch size, grid size, grid size, num_anchors, 5 + num_classes)
    each anchor = [t0, tx, ty, tw, th, c1, c2, ..., cn]
    tx, ty values are calculated relative to the grid cell.
    """
    def __init__(self, data_dir, labels_dir, anchors,
                 num_classes=4, iou_threshold=0.5, file_format='txt', type_dataset='train', convert_to_yolo=True):
        """
        param: data_dir - path to obj.data
        param: labels_dir - path to obj.names
        param: anchors - anchor boxes
        param: num_classses - number of classes (default = 4)
        param: iou_threshold - intersection over union  threshold (default = 0.5)
        param: file_foramt - txt or xml format description files (default = 'txt', available = 'xml')
        param: type_dataset - dataset type (default='train', available = 'validation')
        param: convert_to_yolo - needed if the deiscription files have the format of bounding boxes [x1, y1, x2, y2]
        """

        self.class2tag = {}
        with open(labels_dir, 'r') as f:
            for line in f:
                (val, key) = line.split()
                self.class2tag[key] = val

        self.image_paths = []
        self.box_paths = []
        for tag in self.class2tag:
            for file in os.listdir(data_dir + '/' + tag):
                if file.endswith('.jpg'):
                    self.image_paths.append(data_dir + '/' + tag + '/' + file)
                if file.endswith('.' + file_format):
                    self.box_paths.append(data_dir + '/' + tag + '/' + file)

        # sorting to access values by equivalent files
        self.image_paths = sorted(self.image_paths)
        self.box_paths = sorted(self.box_paths)

        assert len(self.image_paths) == len(self.box_paths)
        assert type_dataset in ['train', 'validation']

        self.s = 13
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_anchors = self.anchors.size(0)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.file_format = file_format
        self.type_dataset = type_dataset
        self.convert_to_yolo = convert_to_yolo

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        if self.file_format == 'xml':
            bboxes, class_labels = self.get_boxes_from_xml(self.box_paths[idx])
        if self.file_format == 'txt':
            bboxes, class_labels = self.get_boxes_from_txt(self.box_paths[idx])

        if self.convert_to_yolo:
            for i, box in enumerate(bboxes):
                bboxes[i] = self.convert_to_yolo_box_params(box, image.shape[1], image.shape[0])

        # creating transformations for training
        if self.type_dataset == 'train':
            transforms = alb.Compose(
                [
                    alb.Resize(self.s * 32, self.s * 32),
                    alb.HorizontalFlip(p=0.5),
                    alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=(-25, 25), p=0.5),
                    alb.RandomBrightnessContrast(p=0.2),
                    alb.Normalize(),
                    alb.pytorch.ToTensorV2()
                ], bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels']))
        # creating transformations for validation
        elif self.type_dataset == 'validation':
            transforms = alb.Compose(
                [
                    alb.Resize(416, 416),
                    alb.Normalize(),
                    alb.pytorch.ToTensorV2()
                ], bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels']))

        transformed = transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = torch.tensor(transformed['bboxes'])
        transformed_class_labels = torch.tensor(transformed['class_labels'])

        # target obj has variants:
        #  1 = if object exist with the best anchor
        # -1 = if object exist with not the best anchor (iou > threshold)
        #  0 = others variants (object not exist)

        target = torch.zeros((self.s, self.s, self.num_anchors, self.num_classes + 5), dtype=torch.float32)

        for i, box in enumerate(transformed_bboxes):
            x_cell = int(self.s * box[0])
            y_cell = int(self.s * box[1])

            calculate_ious = inersection_over_union_anchors(bbox_wh=box[2:4], anchors=self.anchors)
            best_index = torch.argmax(calculate_ious, dim=0)

            tx = self.s * box[0] - x_cell
            ty = self.s * box[1] - y_cell
            tw = torch.log(box[2] / self.anchors[best_index, 0])
            th = torch.log(box[3] / self.anchors[best_index, 1])

            target[y_cell, x_cell, best_index, 1:5] = torch.tensor([tx, ty, tw, th])
            target[y_cell, x_cell, best_index, 5 + transformed_class_labels[i]] = 1

            for index, iou in enumerate(calculate_ious):
                if index == best_index:
                    target[y_cell, x_cell, index, 0] = 1

                elif iou > self.iou_threshold:
                    target[y_cell, x_cell, index, 0] = -1

        return {"image": transformed_image, "target": target}

    def __len__(self):
        return len(self.image_paths)

    def get_boxes_from_txt(self, txt_filename: str):
        boxes = []
        class_labels = []

        with open(txt_filename) as f:
            for obj in f:
                param_list = list(map(float, obj.split()))

                boxes.append(param_list[1:])
                class_labels.append(int(param_list[0]))

        return boxes, class_labels

    def get_boxes_from_xml(self, xml_filename: str):
        boxes = []
        class_labels = []

        with open(xml_filename) as f:
            xml_content = xmltodict.parse(f.read())
        xml_object = xml_content['annotation']['object']

        if type(xml_object) is dict:
            xml_object = [xml_object]

        if type(xml_object) is list:
            for obj in xml_object:
                boxe_list = list(map(float, [obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'],
                                             obj['bndbox']['ymax']]))
                boxes.append(boxe_list)
                class_labels.append(self.class2tag[obj['name']])

        return boxes, class_labels

    def convert_to_yolo_box_params(self, box_coordinates, im_w, im_h):
        ans = list()

        ans.append((box_coordinates[0] + box_coordinates[2]) / 2 / im_w)  # x_center
        ans.append((box_coordinates[1] + box_coordinates[3]) / 2 / im_h)  # y_center

        ans.append((box_coordinates[2] - box_coordinates[0]) / im_w)  # width
        ans.append((box_coordinates[3] - box_coordinates[1]) / im_h)  # height
        return ans
