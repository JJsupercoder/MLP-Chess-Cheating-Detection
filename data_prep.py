import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


with open("train_indices.txt") as f:
    train_indices = set(f.read().strip().split())

with open("test_indices.txt") as f:
    test_indices = set(f.read().strip().split())

with open("val_indices.txt") as f:
    val_indices = set(f.read().strip().split())

train_arrays = []
test_arrays = []
val_arrays = []
for f in os.listdir("samples"):
    if f[:-6] in train_indices:
        arr = np.loadtxt(f"samples/{f}", delimiter=',')
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1,24)
        train_arrays.append(arr)
    elif f[:-6] in test_indices:
        arr = np.loadtxt(f"samples/{f}", delimiter=',')
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1,24)
        test_arrays.append(arr)
    elif f[:-6] in val_indices:
        arr = np.loadtxt(f"samples/{f}", delimiter=',')
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1,24)
        val_arrays.append(arr)


all_train = np.vstack(train_arrays)
all_test = np.vstack(test_arrays)
all_val = np.vstack(val_arrays)

all_train_unique = np.unique(all_train, axis=0)
all_test_unique = np.unique(all_test, axis=0)
all_val_unique = np.unique(all_val, axis=0)

labels_train = all_train_unique[:, 22]
labels_test = all_test_unique[:, 22]
labels_val = all_val_unique[:, 22]

inputs_train = np.delete(all_train_unique, -2, axis=1)
inputs_test = np.delete(all_test_unique, -2, axis=1)
inputs_val = np.delete(all_val_unique, -2, axis=1)

np.savetxt("train.txt", all_train_unique, fmt="%.4e", delimiter=",")
np.savetxt("test.txt", all_test_unique, fmt="%.4e", delimiter=",")
np.savetxt("val.txt", all_val_unique, fmt="%.4e", delimiter=",")
# Initialize the scaler
# scaler = StandardScaler()

# # Fit on training data and transform both training and test data
# X_train = scaler.fit_transform(inputs_train)
# X_test = scaler.transform(inputs_test)
# X_val = scaler.transform(inputs_val)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Compute class weights
# class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels_train)
# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# # Convert to PyTorch tensors
# X_train, X_test, X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)
# y_train, y_test, y_val = torch.tensor(labels_train, dtype=torch.long), torch.tensor(labels_test, dtype=torch.long), torch.tensor(labels_val, dtype=torch.long)
