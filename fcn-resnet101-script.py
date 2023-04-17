import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from torchvision.datasets import DatasetFolder
from dataloader import DataLoaderSegmentation


# the path should contain three folders -> leftlmg8bit and label_processed
path = './data_subset/'

# train and val only
loader = DataLoaderSegmentation(path, "train")


image_datasets = {x: DataLoaderSegmentation(path, x) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
num_epochs = 1
batch_size = 16
learning_rate = 0.001

# Define model
model = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
# Modify model to output correct number of classes
num_classes = 39
#model.classifier = DeepLabHead(2048, num_classes)
model.classifier = nn.Conv2d(2048, num_classes, kernel_size=(1, 1), stride=(1, 1))

# Freeze the backbone layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Train the model
model.to(device)

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataloaders_dict['train']):
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.train()
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)['out']
        loss = criterion(outputs, targets.squeeze(1))
        loss.backward()
        optimizer.step()

        # Print statistics
        if (i + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloaders_dict['train'])}], Loss: {loss.item():.4f}")
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for j, (val_inputs, val_targets) in enumerate(dataloaders_dict['val']):
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                val_outputs = model(val_inputs)['out']
                val_loss += criterion(val_outputs, val_targets.squeeze(1)).item()

            val_loss /= len(dataloaders_dict['val'])

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")

