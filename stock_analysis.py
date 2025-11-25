import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
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


# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.linear(last_out)


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
    """Train the LSTM model and record history"""
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


def process_stock_data(file_path, window_sizes, hidden_size=50, epochs=50):
    """Main pipeline for processing stock data"""
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

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        # 5. Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 6. Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(
            input_size=scaled_data.shape[1],
            hidden_size=hidden_size
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 7. Train the model
        print(f"\nTraining for window size {ws}")
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

    # Process all stock files
    final_results = {}
    for file in stock_files:
        stock_name = file.split('.')[0]
        print(f"\nProcessing {stock_name}...")
        final_results[stock_name] = process_stock_data(file, window_sizes)

    # Compare results
    print("\nFinal Results:")
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
    with open('final_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    print("\nAll results saved to final_results.pkl")