# Stock Data Analysis

A deep learning project for stock price prediction using **LSTM** and **Transformer** models. This project compares the effectiveness of different neural network architectures in forecasting stock opening prices.

## Overview

Stock price prediction is a challenging task due to the high volatility and nonlinear patterns in financial markets. This project explores two deep learning approaches:

- **LSTM (Long Short-Term Memory)**: A recurrent neural network that excels at capturing long-term dependencies in sequential data.
- **Transformer**: An attention-based architecture that can model complex relationships across different time steps.

## Features

- Data preprocessing pipeline for stock market data (handling formats like "1,234.56", "1.2M", "0.5%")
- Time-series windowing with configurable window sizes (10, 20, 50 days)
- MinMax normalization for feature scaling
- Training with validation monitoring (Loss & MAE metrics)
- Model comparison and visualization tools

## Project Structure

```
├── stock_analysis.py            # LSTM-based stock prediction
├── stock_analysis_transformer.py # Transformer-based stock prediction
├── data_evaluation.ipynb        # Model comparison and visualization
├── data/                        # Stock data (CSV files)
│   ├── dax40.csv
│   ├── dji.csv
│   ├── ftse100.csv
│   └── sp500.csv
├── requirements.txt             # Python dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train LSTM Model
```bash
python stock_analysis.py
```

### Train Transformer Model
```bash
python stock_analysis_transformer.py
```

### Compare Results
Open `data_evaluation.ipynb` in Jupyter Notebook to visualize and compare the performance of both models.

## Results

The trained models output:
- `final_results.pkl` - LSTM model results
- `final_results_transformer.pkl` - Transformer model results

These files contain training history, predictions, and scalers for each stock and window size combination.

## Data Format

Input CSV files should contain the following columns:
- `Date`: Trading date
- `Price`, `Open`, `High`, `Low`: Price data (supports comma-separated format)
- `Vol.`: Trading volume (supports M/K/B suffixes)
- `Change %`: Daily price change percentage