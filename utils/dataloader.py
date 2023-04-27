# import sys
# sys.setrecursionlimit(100)

import tensorflow as tf
import numpy as np
import albumentations as A
from pycocotools.coco import COCO
import os
from skimage.io import imread
from skimage.color import gray2rgb
from loguru import logger

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, ann_file, image_dir, batch_size, augmentations = False):
        self.batch_size = batch_size
        self.coco = COCO(ann_file)
        self.augmentations = augmentations
        self.class_names = []
        self.format_class()
        self.image_ids = self.coco.getImgIds()
        np.random.shuffle(self.image_ids)
        self.max_id = 0
        self.image_dir = image_dir
        self.augmentations = augmentations
        
    def format_class(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.max_id = np.array([cat["id"] for cat in cats]).max()
        self.class_names = ["N/A"] * (self.max_id + 1)
        self.class_names[0] = "back"
        for cat in cats:
            self.class_names[cat["id"]] = cat["name"]

    def get_coco_labels(self, img_id, image_shape):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        points = []
        t_class = []
        for a, ann in enumerate(anns):
            bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox'] 
            t_cls = ann["category_id"]
            if bbox_w == 0 and bbox_h == 0:
                x_center = bbox_x
                y_center = bbox_y
            else:
                x_center = bbox_x + (bbox_w / 2)
                y_center = bbox_y + (bbox_h / 2)
            x_center = np.clip(x_center, 0.1, image_shape[1] - 0.1)
            y_center = np.clip(y_center, 0.1, image_shape[0] - 0.1)
            points.append([x_center, y_center])
            t_class.append([t_cls])
        points = np.array(points)
        t_class = np.array(t_class)
        return points.astype(np.float32), t_class.astype(np.float32)

    def get_coco_from_id(self, img_id):
        img = self.coco.loadImgs([img_id])[0]
        file_name = img['file_name']
        image_path = os.path.join(self.image_dir, file_name)
        image = imread(image_path)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)
        if len(image.shape) == 4:
            image = image[...,:3]
        t_points, t_class = self.get_coco_labels(img['id'], image.shape)
        return image, t_points, t_class#, is_crowd

    def pad_points_and_classes(self, points, classes):
        maxobj = max([len(x) for x in points])
        assert len(points) == len(classes), "mismatch of points and classes length list before conversion"
        assert maxobj == max([len(x) for x in classes]), "max obj of points and classes do not match"
        padded_points = np.zeros([len(points), maxobj, 2], dtype=np.float32)
        padded_classes = np.zeros([len(points), maxobj, self.max_id+1], dtype=np.float32)

        for i in range(len(points)):
            padded_points[i, 0:len(points[i])] = points[i]
            padded_classes[i, 0:len(points[i])] = classes[i]

        return padded_points, padded_classes
    
    def keypointsafecrop(self, augmentations, image, t_point, t_class, depth=0, max_depth=5):
        if depth > max_depth:
            logger.debug("max depth reached")
            return self.augmentations(image = image, keypoints = t_point, class_labels=t_class)
        
        transformed = self.augmentations(image = image, keypoints = t_point, class_labels=t_class)
        if len(transformed['keypoints']) == 0:
            #logger.debug("recursive bboxsafecrop")
            return self.keypointsafecrop(augmentations, image, t_point, t_class, depth + 1, max_depth)
        return transformed

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        low = idx * self.batch_size  # image_id starts from 1 in coco format
        high = min(low + self.batch_size, len(self.image_ids))
        images = []
        t_points = []
        t_classes = []
        for i in range(low, high):
            _image_id = self.image_ids[i]
            image, t_point, t_class = self.get_coco_from_id(_image_id)
            if self.augmentations:
                # transformed = self.keypointsafecrop(self.augmentations, image, t_point, t_class)
                transformed = self.augmentations(image = image, keypoints = t_point, class_labels=t_class)
                t_class = np.array(transformed['class_labels'])
                t_point = np.array(transformed['keypoints'])
                image = transformed['image']
            if not len(t_class):
                t_class = np.array([[0]])
                t_point = np.array([[0,0]])
            # using A.compose, paste to padded image divisable by 128 if size is not divisable by 128
            # not using resize to avoid point coord conversion calculation
            if image.shape[0] % 128 != 0 or image.shape[1] % 128 != 0:
                pad = np.ones((int(128 * (image.shape[0] // 128 + 1)), int(128 * (image.shape[1] // 128 + 1)), 3), dtype=np.uint8) * 255
                pad[0:image.shape[0], 0:image.shape[1], :] = image
                image = pad
            #images.append(image/255.)            
            images.append(image)
            t_points.append(t_point)
            t_classes.append(t_class)

        t_points, t_classes = self.pad_points_and_classes(t_points, t_classes)
        images = np.array(images, dtype=np.float32)
        images = np.array(images)

        return images, np.concatenate((t_points, t_classes), axis=-1)

    def on_epoch_end(self):
        np.random.shuffle(self.image_ids)
    
        # return np.array([
        #     resize(imread(file_name), (200, 200))
        #        for file_name in batch_x]), np.array(batch_y)
