# Wczytaj checkpoint
checkpoint = torch.load(filename)

# Załaduj stany
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Kontynuuj od tej epoki
start_epoch = checkpoint['epoch'] + 1
loss = checkpoint['loss']

print(f"Załadowano checkpoint z epoki {start_epoch-1}, loss: {loss:.4f}")