import torch
import torchvision.models as models
import torch.nn as nn
import pickle

label = []
with open('label.pkl', 'rb') as f:
    label = pickle.load(f)

label = torch.tensor(label) # 512, 10, 2
print(label[0])

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.label = label

    def __len__(self):
        return len(self.images)-1

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])

        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label[idx].float()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create dataset
rgb_dir = 'rgb path'
dataset = CustomDataset(rgb_dir, label, transform=transform)

train_dataset = torch.utils.data.Subset(dataset, range(400))
test_dataset = torch.utils.data.Subset(dataset, range(400, 512))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load a pre-trained ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=10 * 2)
model = resnet.to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(15): 
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        yh = model(x)
        loss = criterion(yh.reshape(16,10,2), y)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()  # Set the model to evaluation mode
test_loss = 0
criterion = nn.L1Loss()  # Example using Mean Squared Error for loss calculation

with torch.no_grad(): 
    for idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        yh = model(x)
        loss = criterion(yh.reshape(16,10,2), y)  # Assuming we want to compare reconstructed vs original
        test_loss += loss.item()

        if not idx:
            print(f'Predicted: {yh[0].reshape(10,2)}')
            print(f'Actual: {y[0]}')


average_loss = test_loss / len(test_loader)
print(f'Average Loss: {average_loss:.4f}')

torch.save(model.state_dict(), 'cp.pth')
