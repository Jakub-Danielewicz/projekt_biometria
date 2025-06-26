import matplotlib.pyplot as plt
from data import scan_image_folder, OCRDataset, transforms, OCRDataLoader

if __name__ == '__main__':
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


    dataloader = OCRDataLoader(dataset, batch_size=1, validation_split=0.2, shuffle=True)
    train_loader = dataloader  # to jest OK, bo OCRDataLoader dziedziczy po DataLoader
    val_loader = dataloader.split_validation()  # to jest DataLoader z samplerem walidacyjnym


    # Pick an index to display
    idx = 10
    i = 0
    for images, labels in train_loader:
        i+=1
        image = images[0]
        label = labels[0]
    print(f"Ile jest obrazków: {i}")

    # Display the image and label
    plt.imshow(image.squeeze(0), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")