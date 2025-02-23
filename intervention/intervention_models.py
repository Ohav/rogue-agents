import torch 
from torch import nn
import torch.nn.functional as F
from torch import optim

import numpy as np

MINIMUM_RESET_TURN = 0

def normalize(turn_count, entropy, varentropy, kurtosis):
    turn_count = 1 - np.clip((turn_count // 2 - MINIMUM_RESET_TURN) / 10, 0, 1)
    entropy = ((np.log(entropy) + 66) / 66.42) * 2 - 1
    varentropy = (np.log(varentropy) / 62 + 1) * 2 - 1
    kurtosis = (kurtosis / 6.6) * 2  - 1
    return turn_count, entropy, varentropy, kurtosis

class SmallNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 2)
        self.out = nn.Linear(2, 1)
        

    def forward(self, x):
        x = x[:, :self.input_size].reshape(-1, self.input_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return F.sigmoid(x)

def train(net, train_games, num_epochs=10, learning_rate=0.1, verbose=False):
    # criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5]))  # Use BCEWithLogitsLoss for numerical stability
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) # Reduce LR if val loss plateaus
    train_turns, train_labels = train_games
    last_total_loss = 999 * len(train_turns)

    sample_count = len(train_turns)
    batch_size = 32

    for epoch in range(num_epochs):
        train_loss = 0.0
        for i in range(0, sample_count // batch_size + 1):
            labels = torch.Tensor(train_labels[i*batch_size:(i+1)*batch_size]).float().squeeze()
            features = torch.Tensor(train_turns[i*batch_size:(i+1)*batch_size]).float()
            
            optimizer.zero_grad()
            # Forward pass
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if torch.isnan(loss):
                raise Exception()

            train_loss += loss.item()
            scheduler.step(loss.item())
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        if verbose:
            test_loss = criterion(net(test_turns), test_labels)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_games)}, Test: {test_loss}")

    # net.load_state_dict(best_model)
    # path = f'{save_path}{name}.pth'
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    # torch.save(best_model, path)

    return net

class FullyConnected(nn.Module):
    def __init__(self, feature_indices, normalization_params, degree):
        super().__init__()
        self.feature_indices = feature_indices
        self.normalization_params = normalization_params
        self.polynomial_features = PolynomialFeatures(degree=degree)

    def def_net(self, input_size):
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.out = nn.Linear(20, 1)
        
    def forward(self, x):
        x = self.normalize(x)
        x = torch.Tensor(self.polynomial_features.transform(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return F.sigmoid(x)
    
    def fit(self, train_set):
        train_x, train_y = train_set
        x_norm = self.normalize(train_x)
        x_poly = self.polynomial_features.fit_transform(x_norm)
        self.def_net(x_poly.shape[-1])
        train(self, (train_x, train_y))

    def __call__(self, x):
        return self.forward(x)

    def predict(self, x):
        return self.forward(x)

    def normalize(self, x):
        norm_x = np.zeros_like(x)
        for i in range(len(self.normalization_params)):
            v_min, v_max = self.normalization_params[i]
            norm_x[:, i] = (x[:, i] - v_min) / (v_max - v_min) 
            norm_x[:, i] = norm_x[:, i] * 2 - 1
        norm_x = norm_x[:, self.feature_indices]
        return norm_x

from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifierCV
from sklearn.preprocessing import PolynomialFeatures

class PolynomialModel:
    def __init__(self, feature_indices, normalization_params, degree=2):
        self.feature_indices = feature_indices
        self.polynomial_features = PolynomialFeatures(degree=degree)
        self.model = Ridge(alpha=1)
        # self.model = RidgeClassifier(alpha=1)
        # self.model = RidgeClassifierCV()
        self.normalization_params = normalization_params

    def fit(self, train_set):
        train_x, train_y = train_set
        norm_x = self.normalize(train_x)
        x_poly = self.polynomial_features.fit_transform(norm_x)
        self.model.fit(x_poly, train_y.flatten())

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        norm_x = self.normalize(x)
        x_poly = self.polynomial_features.transform(norm_x)
        res = torch.Tensor(self.model.predict(x_poly))
        return res

    def normalize(self, x):
        norm_x = np.zeros_like(x)
        for i in range(len(self.normalization_params)):
            v_min, v_max = self.normalization_params[i]
            norm_x[:, i] = (x[:, i] - v_min) / (v_max - v_min) 
            norm_x[:, i] = norm_x[:, i] * 2 - 1
        norm_x = norm_x[:, self.feature_indices]
        return norm_x
        