import scipy
import numpy as np

def postprocess(preds):
    # parses single result
    coords, labels = preds[...,:2], preds[...,2:]
    torf = scipy.special.softmax(labels, axis=-1)[...,1] > 0.5
    _idx = np.where(torf)[0]
    filtered_coords = coords[_idx]
    return filtered_coords

def pad(image):
    if image.shape[0] % 128 != 0 or image.shape[1] % 128 != 0:
        pad = np.ones((int(128 * (image.shape[0] // 128 + 1)), int(128 * (image.shape[1] // 128 + 1)), 3), dtype=np.uint8) * 255
        pad[0:image.shape[0], 0:image.shape[1], :] = image
        image = pad
    return image