import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import math
from sklearn.preprocessing import MinMaxScaler


# Custom Dataset class for stock data
class StockDataset(Dataset):
    def __init__(self, data, window_size):
        self.X = []
        self.y = []

        # Create time-series windows
        for i in range(len(data) - window_size):
            self.X.append(data[i:i+window_size, :])  # Input window
            self.y.append(data[i+window_size, 0])    # Predict next Open value

        # Convert to PyTorch tensors
        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Transformer Model for Stock Prediction
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()

        # Input projection: project input features to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layer
        self.fc_out = nn.Linear(d_model, 1)

        self.d_model = d_model

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]

        # Project input to d_model dimensions
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Use the last time step's output for prediction
        x = x[:, -1, :]  # [batch_size, d_model]

        # Output prediction
        return self.fc_out(x)


def read_and_preprocess(file_path):
    """Read and preprocess stock data"""
    df = pd.read_csv(file_path)

    # Convert date and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Clean numerical columns by removing commas
    numeric_columns = ['Price', 'Open', 'High', 'Low']
    for col in numeric_columns:
        df[col] = df[col].str.replace(',', '').astype(float)

    # Clean 'Vol.' column to handle M (millions), K (thousands), and B (billions)
    df['Vol.'] = (
        df['Vol.']
        .replace({'M': 'e6', 'K': 'e3', 'B': 'e9'}, regex=True)
        .astype(float)
    )

    # Clean 'Change %' column
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100

    # Drop rows with NaN or inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df[['Open', 'High', 'Low', 'Vol.', 'Change %']]


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    """Train the Transformer model and record history"""
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_mae = 0.0
        train_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            mae = torch.mean(torch.abs(outputs.squeeze() - targets))
            epoch_train_mae += mae.item()
            train_batches += 1

        avg_train_loss = epoch_train_loss / train_batches
        avg_train_mae = epoch_train_mae / train_batches

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mae = 0.0
        val_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)

                epoch_val_loss += loss.item()
                mae = torch.mean(torch.abs(outputs.squeeze() - targets))
                epoch_val_mae += mae.item()
                val_batches += 1

        avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else float('inf')
        avg_val_mae = epoch_val_mae / val_batches if val_batches > 0 else float('inf')

        history['train_loss'].append(avg_train_loss)
        history['train_mae'].append(avg_train_mae)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}")

    return history


def process_stock_data(file_path, window_sizes, d_model=64, nhead=4, num_layers=2, epochs=50):
    """Main pipeline for processing stock data with Transformer"""
    # 1. Read and preprocess data
    raw_data = read_and_preprocess(file_path).values

    # 2. Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(raw_data)

    # Check for NaN or inf in scaled data
    if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
        raise ValueError("Scaled data contains NaN or inf values.")

    results = {}

    for ws in window_sizes:
        # Ensure the dataset has enough rows for the window size
        if len(scaled_data) <= ws:
            print(f"Skipping window size {ws}: dataset too small ({len(scaled_data)} rows).")
            continue

        # 3. Create dataset with current window size
        dataset = StockDataset(scaled_data, ws)

        # 4. Split into training and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        if train_size <= 0 or test_size <= 0:
            print(f"Skipping window size {ws}: insufficient samples after splitting.")
            continue

        # Time-series split: use first 80% for training, last 20% for testing
        train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

        # 5. Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 6. Initialize Transformer model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TransformerModel(
            input_size=scaled_data.shape[1],
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=d_model * 2,
            dropout=0.1
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 7. Train the model
        print(f"\nTraining Transformer for window size {ws}")
        history = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs=epochs)

        # 8. Collect predictions
        model.eval()
        all_predictions = []
        all_actuals = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                all_predictions.extend(outputs.squeeze().cpu().numpy())
                all_actuals.extend(targets.cpu().numpy())

        # 9. Save the results
        results[ws] = {
            'history': history,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'scaler': scaler
        }

    return results


if __name__ == "__main__":
    # Configuration
    window_sizes = [10, 20, 50]
    stock_files = ['data/dax40.csv', 'data/dji.csv', 'data/ftse100.csv', 'data/sp500.csv']

    # Transformer hyperparameters
    d_model = 64      # Embedding dimension
    nhead = 4         # Number of attention heads
    num_layers = 2    # Number of transformer encoder layers
    epochs = 50

    # Process all stock files
    final_results = {}
    for file in stock_files:
        stock_name = file.split('.')[0]
        print(f"\nProcessing {stock_name} with Transformer...")
        final_results[stock_name] = process_stock_data(
            file, window_sizes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            epochs=epochs
        )

    # Compare results
    print("\n" + "="*60)
    print("Final Results (Transformer Model):")
    print("="*60)
    for stock, res in final_results.items():
        print(f"\n{stock}:")
        if not res:
            print("  No valid window sizes or no data processed.")
            continue
        for ws, result_dict in res.items():
            final_val_loss = result_dict['history']['val_loss'][-1]
            final_val_mae = result_dict['history']['val_mae'][-1]
            print(f"  Window {ws}: Val Loss: {final_val_loss:.4f}, Val MAE: {final_val_mae:.4f}")

    # Save results to file
    with open('final_results_transformer.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    print("\nAll results saved to final_results_transformer.pkl")
