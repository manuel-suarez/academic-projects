import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def evaluate_model(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(torch.randn(10, 3, 64, 64), torch.randint(0, 2, (10, 1)))
    dataloader = DataLoader(dataset, batch_size=2)

    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(2):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

    # Simulate metric (IoU or Accuracy)
    return torch.rand(1).item()
