import torch


def save_checkpoint(model, optimizer, epoch, loss, filename=None):
     checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }
     if filename is None:
          filename = f"checkpoint_epoch{epoch}_loss{loss:.4f}.pt"
     torch.save(checkpoint, filename)
     print(f"Zapisano model do {filename}")



def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    print(f"Załadowano checkpoint z epoki {start_epoch - 1}, loss: {loss:.4f}")
    return model, optimizer, start_epoch, loss
