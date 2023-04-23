import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA
import pandas as pd


mat_fname = 'ECoG_Handpose.mat'
mat_contents = sio.loadmat(mat_fname)
mat_data = mat_contents['y']

X = mat_data[1:61, :]  # ECoG data (channels 2-61)
y = mat_data[61, :].astype(int)  # Paradigm info (channel 62)

X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


encoder = OneHotEncoder()
encoder.fit(np.array([0, 1, 2, 3]).reshape(-1, 1))  # Fit the encoder on the unique label values
y_train = encoder.transform(y_train.reshape(-1, 1)).toarray()
y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()


# Reshape the data into the required input shape for the LSTM model
timesteps = 1  # the number of timesteps
n_features = X_train.shape[1]

X_train = X_train.reshape(-1, timesteps, n_features)
X_test = X_test.reshape(-1, timesteps, n_features)


import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5, device="cpu"):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob).to(self.device)
        self.fc = nn.Linear(hidden_size, num_classes).to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)
    
    def get_hidden_states(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        _, (hidden_states, _) = self.lstm(x, (h0, c0))
        return hidden_states

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")


# Set the number of PCA components
n_pca_components = 40

# Apply PCA to the train and test datasets

X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

pca = PCA(n_components=n_pca_components, whiten=True)
X_train_pca = pca.fit_transform(X_train_2d)
X_test_pca = pca.transform(X_test_2d)

X_train_pca = X_train_pca.reshape(X_train.shape[0], X_train.shape[1], n_pca_components)
X_test_pca = X_test_pca.reshape(X_test.shape[0], X_test.shape[1], n_pca_components)

X_train_pca_tensor = torch.tensor(X_train_pca, dtype=torch.float32).to(device)
X_test_pca_tensor = torch.tensor(X_test_pca, dtype=torch.float32).to(device)


input_size = n_pca_components
# input_size = n_features
hidden_size = 128
num_layers = 2
num_classes = y_train.shape[1]

model = LSTMModel(input_size, hidden_size, num_layers, num_classes, device='cuda').to(device)

# Convert data to PyTorch tensors
# Convert data to PyTorch tensors
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)


# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
losses = []  # Initialize an empty list to store the losses

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_pca_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record the loss
    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions using the trained LSTM model
y_pred_train = model(X_train_pca_tensor).cpu().detach().numpy()
y_pred_test = model(X_test_pca_tensor).cpu().detach().numpy()

# Get the class labels
y_pred_train_labels = np.argmax(y_pred_train, axis=1)
y_pred_test_labels = np.argmax(y_pred_test, axis=1)

y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)


# Calculate the accuracy
train_accuracy = np.mean(y_pred_train_labels == y_train_labels)
test_accuracy = np.mean(y_pred_test_labels == y_test_labels)

print(f"Train accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# Visualize the training loss

import matplotlib.pyplot as plt

# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')

# y_ticks = np.arange(0, np.max(losses), 0.2)
# plt.yticks(y_ticks)
# plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')

