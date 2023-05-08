import numpy as np
import random
from albumentations import DualTransform
import albumentations.augmentations.functional as F
from albumentations.augmentations.crops.functional import crop, crop_keypoint_by_coords

class KeypointSafeRandomCrop(DualTransform):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width

    # def apply(self, img, h_start=0, w_start=0, **params):
    #     return F.RandomCrop(img, self.height, self.width, h_start, w_start)
    def apply(self, img, x_min, x_max, y_min, y_max, **params):
        return crop(img, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    
    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        
        random_keypoint = random.choice(params["keypoints"])
        x, y = random_keypoint[:2]

        min_x = int(max(x - self.width, 0)) +1
        min_y = int(max(y - self.height, 0)) +1
        max_x = int(min(x, img_w - self.width)) -1
        max_y = int(min(y, img_h - self.height)) -1
        if min_x >= max_x:
            x_min = min_x
            x_max = x_min + self.width
        else:
            x_min = random.randint(min_x, max_x)
            x_max = x_min + self.width
        if min_y >= max_y:
            y_min = min_y
            y_max = y_min + self.height
        else:
            y_min = random.randint(min_y, max_y)
            y_max = y_min + self.height
        ret = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        return ret
    
    def apply_to_keypoint(self, keypoint, x_min, x_max, y_min, y_max,  **params):
        return crop_keypoint_by_coords(keypoint,
                                         crop_coords=(x_min, y_min, x_max, y_max))

    @property
    def targets_as_params(self):
        return ["image", "keypoints"]

    def get_transform_init_args_names(self):
        return ("height", "width", "erosion_rate")