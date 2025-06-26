import torch
import torch.optim as optim
from models.mlp import HandwritingMLP
from models.cnn import HandwritingCNN, HandwritingOCRNet

def save_checkpoint(model, optimizer, epoch, loss, LR, filename=None):
     checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'model_type': type(model).__name__,
    'optimizer_type': type(optimizer).__name__,
    'model_kwargs': model.__dict__.get('_kwargs', {}),
    'optimizer_kwargs': {**(optimizer.defaults if hasattr(optimizer, 'defaults') else {}), 'lr': LR},
     }
     if filename is None:
          filename = f"checkpoint_epoch{epoch}_loss{loss:.4f}.pt"
     torch.save(checkpoint, filename)
     print(f"Zapisano model do {filename}")


def load_checkpoint(filename):
    checkpoint = torch.load(filename)

    model_type = checkpoint['model_type']
    optimizer_type = checkpoint['optimizer_type']
    model_kwargs = checkpoint.get('model_kwargs', {})
    optimizer_kwargs = checkpoint.get('optimizer_kwargs', {})

    # Zmapuj nazwę klasy na konstruktor
    model_classes = {
        'HandwritingMLP': HandwritingMLP,
        'HandwritingCNN': HandwritingCNN,
        'HandwritingOCRNet': HandwritingOCRNet
    }
    optimizer_classes = {
        'Adam': optim.Adam,
        'SGD': optim.SGD
    }

    model = model_classes[model_type](**model_kwargs)
    optimizer = optimizer_classes[optimizer_type](model.parameters(), **optimizer_kwargs)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']

    print(f"Załadowano checkpoint z epoki {start_epoch - 1}, loss: {loss:.4f}")
    return model, optimizer, start_epoch, loss
