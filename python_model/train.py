#To jest wzięte żywcem z chata jkc z minimalnymi modyfikacjami
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import HandwritingCNN, HandwritingOCRNet
from models.mlp import  HandwritingMLP

from data import scan_image_folder, OCRDataset, OCRDataLoader
from data.transforms import Compose, Resize, Threshold, ToTensor, Erode, Invert

def validateModel(model, dataloader):
    global device
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct/total

VAL_EVERY = 3
BATCH_SIZE = 8
LR = 0.001
DATA_DIR = ".\python_model\data\set\output_letters_cleaned"

if __name__ == "__main__":
    # Transform pipeline
    transform = Compose([
        Resize((64, 64)),
       # Threshold(190),
       # Erode(),
       # Invert(),
        ToTensor()
    ])


    image_paths, labels = scan_image_folder(DATA_DIR)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(labels)))}
    labels = [class_to_idx[l] for l in labels]

    print(image_paths[0], labels[0])
    dataset = OCRDataset(image_paths, labels, transform=transform)
    #dataset.apply_transform()

    # Tworzenie dataloadera z podziałem na walidację
    dataloader = OCRDataLoader(dataset, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)
    train_loader = dataloader  # to jest OK, bo OCRDataLoader dziedziczy po DataLoader
    val_loader = dataloader.split_validation()  # to jest DataLoader z samplerem walidacyjnym

    # Sprawdź, czy val_loader nie jest None
    if val_loader is None:
        raise ValueError("Validation loader is None! Sprawdź parametr validation_split.")

    if torch.cuda.is_available():
        print("Trenowanie na CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwritingCNN(num_classes=len(set(labels))).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- Trening ---
    for epoch in range(15):
        model.train()
        total_loss = 0
        i = 0
        correct = 0
        for images, labels in train_loader:
           # print("images:", images.shape, images.min().item(), images.max().item())
           # print("labels:", labels[:10])
            #break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            i += BATCH_SIZE
        print(f"Epoch {epoch+1} - loss: {total_loss:.4f} - accuracy: {correct/i * 100:.4f}%")
        if (epoch+1)%VAL_EVERY == 0:
            print(f"Dokładność na zbiorze walidacyjnym: {validateModel(model, val_loader)*100}%")


