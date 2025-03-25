import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from utils import load_data as util_load_data
# Force unbuffered output
sys.stdout.flush()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED = 2025

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=1, num_layers=3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        
        for i in range(num_layers):
            if i == 0:
                layer = nn.Linear(input_size, hidden_size)
                relu = nn.ReLU()
                prev_output_size = layer.out_features
            elif i == num_layers-1:
                layer = nn.Linear(prev_output_size, output_size)
                relu = None
            else:
                layer = nn.Linear(prev_output_size, prev_output_size//4)
                relu = nn.ReLU()
                prev_output_size = layer.out_features
            
            self.layers.append(layer)
            if relu:
                # norm = nn.BatchNorm1d(layer.out_features)
                # self.layers.append(norm)
                self.layers.append(relu)
                dropout = nn.Dropout(0.8)
                self.layers.append(dropout)
        print(self.layers)
        self.sigmoid = nn.Sigmoid()
        
    #     self.reset_params()

    # @staticmethod
    # def init_weights(module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_normal_(module.weight)
    #         nn.init.constant_(module.bias, 0)

    # def reset_params(self):
    #     for module in self.modules():
    #         self.init_weights(module)

    def forward(self, x):
        # out = self.fc1(x)
        # out = self.relu1(out)
        # out = self.fc2(out)
        # out = self.relu2(out)
        # out = self.fc3(out)
        out = self.layers(x)
        out = self.sigmoid(out)
        return out

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def main(self):
        # Load Data
        train_loader, val_loader = self.load_data()

        # Model, Loss & Optimizer
        input_size = train_loader.dataset.tensors[0].shape[1]
        hidden_size = 512
        num_layers = 3
        output_size = 1
        model = MLP(input_size, hidden_size, output_size, num_layers).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        # Training Loop
        num_epochs = 20
        best_val_loss = float("inf")
        early_stopping_counter = 0
        early_stopping_patience = 5

        for epoch in range(num_epochs):
            train_loss, model, criterion, optimizer = self.train(train_loader, model, criterion, optimizer)
            val_loss, accuracy, f1, report, confusion = self.validate(val_loader, model, criterion)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f},  F1: {f1:.4f}")
            print(f"Classification Report:\n{report}")
            print(f"Confusion Matrix:\n{confusion}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                torch.save(model.state_dict(), "cache/best_mlp.pth")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

        print("Training complete. Best model saved as best_mlp.pth.")

    def load_data(self, batch_size=128):
        X_train, X_test, y_train, y_test = util_load_data()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(self.device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader

    def train(self, train_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, "Train:"):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        return train_loss, model, criterion, optimizer

    def validate(self, val_loader, model, criterion):
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, "Val:"):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        report = classification_report(all_labels, all_preds)
        confusion = confusion_matrix(all_labels, all_preds)

        return val_loss, accuracy, f1, report, confusion


if __name__ == "__main__":
    trainer = Trainer()
    trainer.main()
