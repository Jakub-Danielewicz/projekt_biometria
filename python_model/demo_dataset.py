import matplotlib.pyplot as plt
from data import scan_image_folder, OCRDataset, transforms

data_dir = "data/output_letters"

# Scan the directory and get image paths and labels
image_paths, labels = scan_image_folder(data_dir)

# Compose your transform pipeline
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Threshold(100),
])

# Create the dataset (no transforms for now)
dataset = OCRDataset(image_paths, labels, transform=transform)

# Pick an index to display
idx = 15
image, label = dataset[idx]

# Display the image and label
plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()