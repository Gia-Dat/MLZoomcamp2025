import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# 1. ARCHITECTURE DEFINITION


class AnimalClassifierMobileNet(nn.Module):
    def __init__(self, size_inner=100, droprate=0.0, num_classes=10):
        super(AnimalClassifierMobileNet, self).__init__()
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.classifier = nn.Identity()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(droprate)
        self.output_layer = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# 2. DATA LOADING LOGIC


class AnimalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 3. TRAINING FUNCTION


def train_model(train_dir, val_dir, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_ds = AnimalDataset(train_dir, transform=transform)
    val_ds = AnimalDataset(val_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Initialize Model (using your best found parameters)
    model = AnimalClassifierMobileNet(
        size_inner=100, droprate=0.0, num_classes=10)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} complete.")

    return model


# 4. EXECUTION AND SAVING
if __name__ == "__main__":
    TRAIN_PATH = r'C:\Users\GIA DAT\ML Zoomcamp\11. Capstone Project\data\animal_subset\train'
    VAL_PATH = r'C:\Users\GIA DAT\ML Zoomcamp\11. Capstone Project\data\animal_subset\validation'
    OUTPUT_FILE = "animal_model.pth"

    # Train
    model = train_model(TRAIN_PATH, VAL_PATH)

    # Save the State Dict (Standard way for PyTorch)
    torch.save(model.state_dict(), OUTPUT_FILE)
    print(f"Model saved to {OUTPUT_FILE}")
