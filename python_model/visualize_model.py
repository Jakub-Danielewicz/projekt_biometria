import torch
import os
from dotenv import load_dotenv
from model_saveloader import load_checkpoint
from data import scan_image_folder, OCRDataset, OCRDataLoader
from data.transforms import Compose, Resize, ToTensor
from visualizer import visualize_predictions

# Wczytaj zmienne środowiskowe
load_dotenv()
MODEL = os.getenv("MODEL")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
DATA_DIR = os.getenv("DATA_DIR")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint.pt")  #


val_transform = Compose([
    Resize((64, 64)),
    ToTensor()
])


image_paths, labels = scan_image_folder(DATA_DIR)
setoflabels = sorted(set(labels))
class_to_idx = {cls: idx for idx, cls in enumerate(setoflabels)}
labels = [class_to_idx[l] for l in labels]
dataset = OCRDataset(image_paths, labels, transform=val_transform)
dataloader = OCRDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, optimizer, start_epoch, loss = load_checkpoint(CHECKPOINT_PATH)
model = model.to(device)


visualize_predictions(model, dataloader, class_names=setoflabels, device=device, num_images=6)