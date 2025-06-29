import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import HandwritingCNN, HandwritingOCRNet
from models.mlp import  HandwritingMLP

from data import scan_image_folder, OCRDataset, OCRDataLoader
from data.transforms import Compose, Resize, Threshold, ToTensor, Erode, Invert, RandomShift, RandomRotate
from dotenv import load_dotenv
import os

load_dotenv()
MODEL = os.getenv("MODEL")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
LR = float(os.getenv("LEARNING_RATE"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))
VAL_EVERY = int(os.getenv("VAL_EVERY"))
DATA_DIR = os.getenv("DATA_DIR")

from model_saveloader import save_checkpoint, load_checkpoint
from visualizer import visualize_predictions

def validateModel(model, dataloader):   
    global device, setoflabels, dataset
    dataset.transform = val_transform
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            if len(images) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct/total

if __name__ == "__main__":
    # Transform pipeline
    train_transform = Compose([
    Resize((64, 64)),
    RandomRotate(angle=10),
    RandomShift(max_shift=4),
    ToTensor()
    ])

    val_transform = Compose([
        Resize((64, 64)),
        ToTensor()
    ])


    image_paths, labels = scan_image_folder(DATA_DIR)
    setoflabels = sorted(set(labels))
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(labels)))}
    labels = [class_to_idx[l] for l in labels]

    print(image_paths[0], labels[0])
    dataset = OCRDataset(image_paths, labels, transform=train_transform)

    dataloader = OCRDataLoader(dataset, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)
    train_loader = dataloader 
    val_loader = dataloader.split_validation()  

    if val_loader is None:
        raise ValueError("Validation loader is None! Sprawdź parametr validation_split.")

    if torch.cuda.is_available():
        print("Trenowanie na CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODEL == "HandwritingMLP":
        model = HandwritingMLP(num_classes=len(set(labels))).to(device)
    elif MODEL == "HandwritingCNN":
        model = HandwritingCNN(num_classes=len(set(labels))).to(device)
    elif MODEL == "HandwritingOCRNet":
        model = HandwritingOCRNet(num_classes=len(set(labels))).to(device)
    else:
        raise ValueError(f"Nieznany model: {MODEL}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        dataset.transform = train_transform
        model.train()
        total_loss = 0
        i = 0
        correct = 0
        for images, labels in train_loader:
            if len(images) == 0:
                continue
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

    setattr(model, '_kwargs', {'num_classes': len(setoflabels)})
    save_checkpoint(model, optimizer, epoch, total_loss, LR, "checkp_2.pt")

    visualize_predictions(model, val_loader, class_names = setoflabels, device=device, num_images=6)
 
