#To jest wzięte żywcem z chata jkc z minimalnymi modyfikacjami
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
from PIL import Image

#chciałam to zaimportować ale się poddałam

class Transform:
    def __call__(self, image):
        raise NotImplementedError

class Resize(Transform):
    def __init__(self, size):
        self.size = size  # (width, height)
    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        return cv2.resize(image, self.size)

class Threshold(Transform):
    def __init__(self, thresh=190):
        self.thresh = thresh
    def __call__(self, image):
        _, img = cv2.threshold(image, self.thresh, 255, cv2.THRESH_BINARY)
        return img

class ToTensor(Transform):
    def __init__(self):
        from torchvision.transforms import ToTensor as TorchvisionToTensor
        self.torchvision_to_tensor = TorchvisionToTensor()
    def __call__(self, image):
        if isinstance(image, np.ndarray):
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

transform = Compose([
    Resize((64, 64)),
    Threshold(190),
    ToTensor()
])

# --- CustomImageFolder z transformacją ---

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = transform(image)
        return image, label

# --- Ładowanie danych i podział ---

dataset = CustomImageFolder(root='python_model/data/output_letters_cleaned')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Model CNN ---

class HandwritingCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandwritingCNN(num_classes=len(dataset.classes)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Trening ---

for epoch in range(15):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - loss: {total_loss:.4f}")

# --- Testowanie ---

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Dokładność na zbiorze testowym: {100 * correct / total:.2f}%")
