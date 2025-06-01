import matplotlib.pyplot as plt
from data import scan_image_folder, OCRDataset, transforms

data_dir = ".\python_model\data\set\output_letters_cleaned"

# Scan the directory and get image paths and labels
image_paths, labels = scan_image_folder(data_dir)

# Compose your transform pipeline
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Threshold(200),
    transforms.ToTensor()
])

# Create the dataset (no transforms for now)
dataset = OCRDataset(image_paths, labels, transform=transform)
dataset.apply_transform()

# Pick an index to display
idx = 100
image, label = dataset[idx]

# Display the image and label
plt.imshow(image.squeeze(0), cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()
print(f"Image shape: {image.shape}")
print(f"Label: {label}")