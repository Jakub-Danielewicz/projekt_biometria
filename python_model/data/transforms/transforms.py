import cv2
import numpy as np
from torchvision.transforms import ToTensor as TorchvisionToTensor
import random

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

class ToTensor(Transform):
    def __init__(self):
        self.torchvision_to_tensor = TorchvisionToTensor()

    def __call__(self, image):
        # If image is a NumPy array, convert to PIL Image first
        if isinstance(image, np.ndarray):
            from PIL import Image
            if image.ndim == 2:
                image = Image.fromarray(image, mode='L')
            else:
                image = Image.fromarray(image)
        return self.torchvision_to_tensor(image)
    
class Compose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class Erode(Transform):
    def __init__(self, kernel_size=2):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    def __call__(self, image):
        import cv2
        return cv2.erode(image, self.kernel, iterations=1)

class Invert(Transform):
    def __call__(self, image):
        return 255 - image

class RandomRotate(Transform):
    def __init__(self, angle=10):
        self.angle = angle
    def __call__(self, image):
        h, w = image.shape[:2]
        angle = random.uniform(-self.angle, self.angle)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(image, M, (w, h), borderValue=255)

class RandomShift(Transform):
    def __init__(self, max_shift=4):
        self.max_shift = max_shift
    def __call__(self, image):
        h, w = image.shape[:2]
        tx = random.randint(-self.max_shift, self.max_shift)
        ty = random.randint(-self.max_shift, self.max_shift)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (w, h), borderValue=255)


