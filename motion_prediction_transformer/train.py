import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerModel
from torch.utils.data import Dataset, DataLoader
import numpy as np

input_size = 8  # (x, y, z, yaw, vx, vy, vz, v_yaw)
hidden_size = 128
output_size = 8  # (x, y, z, yaw, vx, vy, vz, v_yaw)
num_layers = 6
nhead = 8

model = TransformerModel(input_size, hidden_size, output_size, num_layers, nhead)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model, 'model.pth')

class MotionDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        past_states = self.data[idx:idx+self.seq_length, :8]
        future_states = self.data[idx+self.seq_length, :8]
        return past_states.clone().detach(), future_states.clone().detach()

# Load data from CSV file
data = np.loadtxt('robot_motion_data.csv', delimiter=',', skiprows=1)
data = torch.tensor(data, dtype=torch.float32)
print("Data Loaded")

seq_length = 100
dataset = MotionDataset(data, seq_length)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Reshape inputs and targets
            inputs = inputs.permute(1, 0, 2)
            targets = targets.unsqueeze(0)
            
            # Prepare the target sequence with an additional dimension for the Transformer
            tgt = torch.zeros_like(targets)
            
            outputs = model(inputs, tgt)
            loss = criterion(outputs.squeeze(0), targets.squeeze(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model, 'transformer_model.pth')

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 500

train_model(model, train_loader, criterion, optimizer, num_epochs)