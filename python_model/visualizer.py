import torch
import torch
#import torch.nn as nn
#import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_saveloader import save_checkpoint, load_checkpoint
from data import scan_image_folder, OCRDataset, OCRDataLoader
from data.transforms import Compose, Resize, ToTensor


def visualize_predictions(model, dataloader, class_names, device, num_images=6):
    model.eval()
    images_shown = 0
    fig = plt.figure(figsize=(15, 6))

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break

                image = images[i].cpu().squeeze().numpy()
                pred_label = class_names[preds[i].item()]
                true_label = class_names[labels[i].item()]
                confidence = probs[i][preds[i]].item()

                ax = fig.add_subplot(2, (num_images + 1)//2, images_shown + 1)
                ax.imshow(image, cmap='gray')
                ax.set_title(f"Przewidywanie modelu: {pred_label} ({confidence*100:.1f}%)\nPrawda: {true_label}")
                ax.axis("off")

                images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.show()



# inicjalizacja dataloadera (TO jest ważne)
#dataloader = OCRDataLoader(dataset, batch_size=16, shuffle=True)

# teraz możesz przekazać go do funkcji:
#visualize_predictions(model, dataloader, class_names=class_names, device=device, num_images=6)




