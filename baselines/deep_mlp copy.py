import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hiddens_size3, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.fc1 = nn.Linear(4, 2)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(hidden_size2, hiddens_size3)
        # self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.fc2(out)
        # out = self.relu2(out)
        # out = self.fc3(out)
        # out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

def train_mlp():
    df = pd.read_csv("mlp/train_val.csv")
    X = df[['depth_1', 'depth_2', 'depth_3', 'depth_4', 'depth_5',
            'depth_6', 'depth_7', 'depth_8', 'depth_9', 'depth_10', 'depth_11',
            'depth_12', 'depth_13', 'depth_14', 'depth_15', 'depth_16', 'depth_17',
            'depth_18', 'depth_19', 'depth_20', 'best_move_eval', 'player_rating',
            'sigmoid_eval_ratio']]
    X = df[['depth_20', 'best_move_eval', 'sigmoid_eval_ratio', 'player_rating']]
    Y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    # Create TensorDataset and DataLoader for shuffling and batching
    batch_size = 4096
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #shuffle = true

    # Define model, loss, and optimizer
    input_size = X_train_tensor.shape[1]
    hidden_size1 = 512
    hidden_size2 = 256
    hidden_size3 = 128
    hidden_size4 = 64
    hidden_size5 = 32
    output_size = 1
    model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader: # iterate through data loader.
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = (y_pred_tensor.cpu().numpy() > 0.5).astype(int)
        y_test_np = y_test_tensor.cpu().numpy()

    accuracy = accuracy_score(y_test_np, y_pred)
    report = classification_report(y_test_np, y_pred)
    confusion = confusion_matrix(y_test_np, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{confusion}")


if __name__ == '__main__':
    train_mlp()