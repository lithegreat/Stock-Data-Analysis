import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        # Fully connected layer for output
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use only the last time step's output
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
        .replace({'M': 'e6', 'K': 'e3', 'B': 'e9'}, regex=True)  # Replace suffixes
        .astype(float)  # Convert to float
    )
    
    # Clean 'Change %' column
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100
    
    # Drop rows with NaN or inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df[['Open', 'High', 'Low', 'Vol.', 'Change %']]

def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    """Train the LSTM model"""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs.squeeze(), targets)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print loss every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on test data"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def process_stock_data(file_path, window_sizes):
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
        # 3. Create dataset with current window size
        dataset = StockDataset(scaled_data, ws)
        
        # 4. Split into training and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])
        
        # 5. Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 6. Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(
            input_size=scaled_data.shape[1],
            hidden_size=50
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Add learning rate
        
        # 7. Train the model
        print(f"\nTraining for window size {ws}")
        train_model(model, train_loader, criterion, optimizer, device, epochs=50)
        
        # 8. Evaluate the model
        test_loss = evaluate_model(model, test_loader, criterion, device)
        results[ws] = test_loss
    
    return results

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
    for ws, loss in res.items():
        print(f"Window {ws}: Test Loss: {loss:.4f}")