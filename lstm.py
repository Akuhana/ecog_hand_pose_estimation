import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

mat_fname = 'ECoG_Handpose.mat'
mat_contents = sio.loadmat(mat_fname)
mat_data = mat_contents['y']

X = mat_data[1:61, :100000]  # ECoG data (channels 2-61)
y = mat_data[61, :100000].astype(int)  # Paradigm info (channel 62)

X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Scaling the y values
scaler = MinMaxScaler()
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

scaler.fit(y_train)
y_train = scaler.transform(y_train).flatten()
y_test = scaler.transform(y_test).flatten()

encoder = LabelEncoder()
encoder.fit([0, 1, 2, 3])
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

unique_y_train_values = np.unique(y_train)
unique_y_test_values = np.unique(y_test)

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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
        self.dropout = nn.Dropout(dropout_prob).to(self.device)
        self.fc = nn.Linear(hidden_size, num_classes).to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)  # Apply dropout after the LSTM layer
        out = self.fc(out[:, -1, :])
        return out
    
    def get_hidden_states(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        _, (hidden_states, _) = self.lstm(x, (h0, c0))
        return hidden_states

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

input_size = n_features
hidden_size = 256
num_layers = 2
num_classes = len(np.unique(y))

model = LSTMModel(input_size, hidden_size, num_layers, num_classes, device='cuda').to(device)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []  # Initialize an empty list to store the losses

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record the loss
    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Visualize the training loss

import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

y_ticks = np.arange(0, np.max(losses), 0.2)
plt.yticks(y_ticks)
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')


# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Extract the hidden states from the LSTM model
hidden_states_train = model.get_hidden_states(X_train_tensor)[-1].detach().cpu().numpy()
hidden_states_test = model.get_hidden_states(X_test_tensor)[-1].detach().cpu().numpy()

# Train the LinearDiscriminantAnalysis model using the extracted hidden states
print(hidden_states_train.shape, y_train.shape)
lda = LinearDiscriminantAnalysis()
lda.fit(hidden_states_train, y_train)

# Evaluate the LDA model using the test dataset
lda_accuracy = lda.score(hidden_states_test, y_test)
print("LDA accuracy: {:.2f}%".format(lda_accuracy * 100))

# # Hyperparameter tuning

# import ray
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune.search.skopt import SkOptSearch
# from skopt.space import Integer, Real

# ray.init(num_cpus=16, num_gpus=1)

# def train_lstm(config):
        
#     model = LSTMModel(config["input_size"], config["hidden_size"], config["num_layers"], config["num_classes"],
#                       config["dropout_prob"], device=device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

#     for epoch in range(config["num_epochs"]):
#         inputs, labels = X_train_tensor.to(device), y_train_tensor.to(device)
#         outputs = model(X_train_tensor)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     hidden_states_train = model.get_hidden_states(X_train_tensor.to(device))
#     hidden_states_test = model.get_hidden_states(X_test_tensor.to(device))

#     hidden_states_train = hidden_states_train[-1].cpu().detach().numpy()
#     hidden_states_test = hidden_states_test[-1].cpu().detach().numpy()

#     lda = LinearDiscriminantAnalysis()
#     lda.fit(hidden_states_train, y_train)

#     transformed_hidden_states_train = lda.transform(hidden_states_train)
#     transformed_hidden_states_test = lda.transform(hidden_states_test)
                                                   
#     lda_score = lda.score(transformed_hidden_states_test, y_test)
#     tune.report(lda_score=lda_score)
    
# search_space = {
#     "input_size": tune.choice([n_features]),
#     "hidden_size": tune.randint(64, 256),
#     "num_layers": tune.randint(1, 3),
#     "dropout_prob": tune.uniform(0.1, 0.5),
#     "lr": tune.loguniform(1e-4, 1e-2),
#     "num_epochs": tune.randint(50, 200),
# }


# skopt_search = SkOptSearch(space=search_space, metric="lda_score", mode="max")

# scheduler = ASHAScheduler(
#     metric="lda_score",
#     mode="max",
#     max_t=2000,
#     grace_period=10,
#     reduction_factor=2
# )

# analysis = tune.run(
#     train_lstm,
#     config=search_space,
#     scheduler=scheduler,
#     num_samples=50,
#     resources_per_trial={"cpu": 1, "gpu": 1},
#     local_dir="./ray_results",
#     name="lstm_hyperopt"
# )



# best_trial = analysis.get_best_trial("lda_score", "max", "last")
# print("Best trial config: {}".format(best_trial.config))
# print("Best trial final validation LDA score: {}".format(best_trial.last_result["lda_score"]))
