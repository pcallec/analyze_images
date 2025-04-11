"""basic_checkpoint_pytorch.py

Demonstrates training a simple CNN on FashionMNIST with checkpointing.

Source:
* https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
* https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
from pathlib import Path 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# ------------------------------
# Step 1: Load and preprocess data
# ------------------------------

# Define transformation: convert images to tensors and normalize
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])


# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# ------------------------------
# Step 2: Define the CNN model
# ------------------------------

# A simple CNN for image classification
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate the model
model = GarmentClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Set up the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# ------------------------------
# Step 3: Training loop for a single epoch
# ------------------------------

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


# ------------------------------
# Step 4: Training and checkpointing
# ------------------------------

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_start = 0

EPOCHS = 6

best_vloss = float("inf")

folder_path = "/home/pcallec/analyze_images/results/basic_checkpoint_pytorch/"
checkpoint_path = Path(folder_path) / "checkpoint.pth"
backup_checkpoint_path = Path(folder_path) / "checkpoint_backup.pth"
checkpoint_interval = 2



Path(folder_path).mkdir(parents=True, exist_ok=True)

if checkpoint_path.exists():
    try:
        checkpoint = torch.load(checkpoint_path)
    except Exception as e:
        print(f"Failed to load main checkpoint: {e}")
        if backup_checkpoint_path.exists():
            try:
                checkpoint = torch.load(backup_checkpoint_path)
                print(f"Loaded backup checkpoint from {backup_checkpoint_path}")
            except Exception as e2:
                print(f"Backup checkpoint also failed: {e2}")
                checkpoint = None
        else:
            print("No backup checkpoint found.")
            checkpoint = None
    
    if checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_vloss = checkpoint.get("best_val_loss", float("inf"))
            epoch_start = checkpoint.get("epoch", -1) + 1
            print(f"Resuming training from epoch {epoch_start}")
        except Exception as e:
            print(f"Checkpoint was found but could not be loaded correctly: {e}")


# Train for specified number of epochs
for epoch in range(epoch_start, EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    
    
    if (epoch + 1) % checkpoint_interval == 0:
        # Create backup checkpoint if checkpoint exists
        if checkpoint_path.exists():
            checkpoint_path.replace(backup_checkpoint_path)

        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_vloss,
                    }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1} at {checkpoint_path}")

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_model_filename = f"best_model_{timestamp}_{epoch}.pth"
        best_model_path = Path(folder_path) / best_model_filename
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch + 1} at {best_model_path}")