import os
from torch.utils.data import Dataset
from .fileloader import load_image_cv2

class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, apply_transform=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.transformed_images = []
        if apply_transform:
            self.apply_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.transformed_images:
            image = self.transformed_images[idx]
        else:
            image = load_image_cv2(self.image_paths[idx])
            if self.transform:
                image = self.transform(image)
        label = self.labels[idx]
        
        return image, label
    def apply_transform(self):
        """
        Apply the transform to all images in the dataset.
        """
        for i in range(len(self.image_paths)):
            image = load_image_cv2(self.image_paths[i])
            if self.transform:
                image = self.transform(image)
            self.transformed_images.append(image)
    def flush(self):
        """
        Clear the transformed images cache.
        """
        self.transformed_images = []