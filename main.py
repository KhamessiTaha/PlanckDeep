from src.dataset import CMBDataset
from src.model import CMBClassifier
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 32
EPOCHS = 5

dataset = CMBDataset("data/cmb_patches.npy", "data/cmb_labels.npy")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CMBClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/cmb_classifier.pt")
