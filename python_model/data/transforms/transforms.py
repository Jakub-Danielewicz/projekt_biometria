import cv2
import numpy as np

class Transform:
    """Base class for all transforms."""
    def __call__(self, image):
        raise NotImplementedError

class Resize(Transform):
    def __init__(self, size):
        self.size = size  # (width, height)

    def __call__(self, image):
        return cv2.resize(image, self.size)

class Threshold(Transform):
    def __init__(self, thresh=127):
        self.thresh = thresh

    def __call__(self, image):
        _, img = cv2.threshold(image, self.thresh, 255, cv2.THRESH_BINARY)
        return img

class ToNumpy(Transform):
    def __call__(self, image):
        return np.array(image)

class Compose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image