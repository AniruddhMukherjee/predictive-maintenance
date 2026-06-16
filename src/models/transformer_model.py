import torch
import torch.nn as nn
import numpy as np
import os
import math

class PositionalEncoding(nn.Module):
    """
    as transformers have no sense of order, positional encoding injects
    cycle postion into the input so that the model can diffrentiate between step1 vs step30.
    """
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)

        # create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices
        pe = pe.unsqueeze(0)  # add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add postional encoding to input embeddings
        x = x + self.pe[:, :x.size(1)]
        return self.dropout_layer(x)
    
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        """
        input_size - no. of features per timestamp
        d_model - intenal model dimension (all attention happens here)
        nhead - number of attention heads (each focuses on a different pattern)
        num_layers - stacked transformer encoder layers
        dim_feedforward - size of feedforward network inside each layer
        dropout - regularization (prevents overfitting)
        """
        super(TransformerModel, self).__init__()

        #project input features into d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)

        # Add positional information
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Core transformer encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True, # input: (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        #output head - map transformer output to single RUL value
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32,1),
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)

        #project to d_model
        x = self.input_projection(x) # (batch, seq, d_model)

        #add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer encoder
        x = self.transformer(x)  # (batch, seq, d_model)

        #Global average pooling across time dimension
        # instead of last timestep, avg all timestep outputs
        x = x.mean(dim=1)      # (batch, d_model)

        #Final Projection
        out = self.fc(x)
        return out.squeeze()
    
def train(train_df, sequence_length=30, epochs=60, batch_size=256, lr=0.001):
    """Training loop for transformer - same as LSTM"""

    # Reuse sequence creation from LSTM
    from models.lstm_model import create_sequence, create_test_sequences

    print("Creating sequence...")
    X, y = create_sequence(train_df, sequence_length)
    print(f"Training sequence: {X.shape}")

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X.shape[2]
    model = TransformerModel(input_size=input_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # ReduceLROnPlatue - automatically lower LR when loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )
    criterion = nn.MSELoss()

    print("Training Trnaformer...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            # Gradient clipping - prevents exploding gradients in transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epoch} - Loss: {avg_loss:.4f}")

    return model

def predict_test(model, test_df, sequence_length=30):
    """Generate preditions for test engines"""
    from models.lstm_model import create_test_sequences

    X_test = create_test_sequences(test_df, sequence_length)
    X_tensor = torch.FloatTensor(X_test)

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    predictions = np.clip(predictions, 0, 125)
    return predictions

def save_model(model, path='models/transformer_rul.pt'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

