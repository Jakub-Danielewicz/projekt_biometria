#To jest wzięte żywcem z chata jkc z minimalnymi modyfikacjami
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import HandwritingCNN

from data import scan_image_folder, OCRDataset, OCRDataLoader
from data.transforms import Compose, Resize, Threshold, ToTensor


if __name__ == "__main__":
    # Transform pipeline
    transform = Compose([
        Resize((64, 64)),
        Threshold(190),
        ToTensor()
    ])

    # Ładowanie danych
    data_dir = ".\python_model\data\set\output_letters_cleaned"
    image_paths, labels = scan_image_folder(data_dir)

    print(image_paths[0], labels[0])
    dataset = OCRDataset(image_paths, labels, transform=transform)
    #dataset.apply_transform()

    # Tworzenie dataloadera z podziałem na walidację
    dataloader = OCRDataLoader(dataset, batch_size=32, validation_split=0.2, shuffle=True)
    train_loader = dataloader  # to jest OK, bo OCRDataLoader dziedziczy po DataLoader
    val_loader = dataloader.split_validation()  # to jest DataLoader z samplerem walidacyjnym

    # Sprawdź, czy val_loader nie jest None
    if val_loader is None:
        raise ValueError("Validation loader is None! Sprawdź parametr validation_split.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwritingCNN(num_classes=len(set(labels))).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # --- Trening ---
    for epoch in range(15):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            print(type(images), type(labels))
            print(labels)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - loss: {total_loss:.4f}")

    # --- Walidacja/Test ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            print(type(images), type(labels))
            print(images)
            break 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Dokładność na zbiorze walidacyjnym: {100 * correct / total:.2f}%")
