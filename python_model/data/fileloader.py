import os
from typing import List, Tuple
import os
import cv2
from typing import List, Tuple

def load_image_cv2(image_path: str, color: str = 'grayscale'):
    """
    Loads an image from a file path using OpenCV.
    Args:
        image_path (str): Path to the image file.
        color (str): 'grayscale' or 'color'.
    Returns:
        image (np.ndarray): Loaded image.
    """
    if color == 'grayscale':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def scan_image_folder(root_dir: str, allowed_exts: Tuple[str] = ('.png', '.jpg', '.jpeg')) -> Tuple[List[str], List[str]]:
    """
    Scans a directory tree and returns image paths and labels (parent folder names).
    """
    image_paths = []
    labels = []
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(allowed_exts):
                image_paths.append(os.path.join(label_dir, fname))
                labels.append(label)
    return image_paths, labels