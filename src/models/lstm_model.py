import torch
import torch.nn as nn
import numpy as np
import os

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        input_size - no. of features per timestep
        hidden_size - no. of LSTM units (memory capacity)
        num_layers - stacked LSTM layers (deeper = more complex patterns)
        droupout - randomly zero out neurons during training to reduce overfitting
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Core LSTM processed seq. of sensor readings over time
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # input shape: (batch, sequence, features)
            dropout=dropout, 
        )

        # Fully connected output layer - maps LSTM output to single RUL value
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # single predicted layer
        )

    def forward(self, x):
        #x shape: (batch_size, squence_length, input_size)

        # Initialize hidden and cell states to 0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take only the last timestamp output (RUL prediction)
        out = self.fc(out[:, -1, :])
        return out.squeeze()
        
def create_sequence(df, sequence_length=30):
    """
    Convert flat dataframe into sequence for LSTM.

    for each engine, slide a window of 'sequnce_length' cycles.
    Each window becomes one training sample
    label = RUL at the LAST cycle of the window

    ex: sequence_length=30
    Cycles 1-30 - label = RUL at cycle 30
    Cycles 2-31 - label = RUL at cycle 31, and so on...
    """
    feature_cols = [c for c in df.columns if c not in ['engine_id', 'cycle', 'RUL']]
    X, y = [], []

    for engine_id in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == engine_id].reset_index(drop=True)
        features = engine_df[feature_cols].values
        labels = engine_df['RUL'].values

        #Slide window across engine's lifecycle
        for i in range(len(engine_df) - sequence_length + 1):
            X.append(features[i:i + sequence_length]) # sequence of readings
            y.append(labels[i + sequence_length - 1]) # RUL at the end of window

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
def create_test_sequences(df, sequence_length=30):
    """
    For test data: take the LAST 'sequence_length' cycles per engine
    That is my prediction window, before our cutoff
    """
    feature_cols = [c for c in df.columns if c not in ['engine_id', 'cycle', 'RUL']]
    X = []

    for engine_id in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == engine_id].reset_index(drop=True)
        features = engine_df[feature_cols].values

        if len(engine_df) >= sequence_length:
            # take last sequence_length cycles
            X.append(features[-sequence_length:])
        else:
            # engine has fewer cycles than window - pad with 0 at start
            pad = np.zeros((sequence_length - len(engine_df), features.shape[1]))
            X.append(np.vstack([pad, features]))

    return np.array(X, dtype=np.float32)

def train(train_df, sequence_length=30, epochs=50, batch_size=256, lr=0.001):
    """LSTM training"""

    print("Creating sequence")
    X, y = create_sequence(train_df, sequence_length)
    print(f"Training sequences: {X.shape}")  #(num_samples, seq_len, num_features)

    # convert to Pytorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #Build model
    input_size = X.shape[2]  # number of features
    model = LSTMModel(input_size=input_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() #minimize mean squared error

    print("Training LSTM...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()   # clear gradients
            predictions = model(X_batch)  # forward pass
            loss = criterion(predictions, y_batch) # compute loss
            loss.backward() # back propagation
            optimizer.step() # update weights
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4}")

    return model

def predict_test(model, test_df, sequence_length=30):
    """Generate RUL predictions for test engine"""
    X_test = create_test_sequences(test_df, sequence_length)
    X_tensor = torch.FloatTensor(X_test)

    model.eval()  # no dropout for inference
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    predictions = np.clip(predictions, 0, 125)
    return predictions

def save_model(model, path='models/lstm_rul.pt'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
