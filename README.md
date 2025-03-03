In recent years, stock price prediction has attracted significant attention in financial research due to its potential impact on investment strategies and risk management. Traditional statistical models often struggle to capture the high volatility and nonlinear patterns in stock market data. In this case, Long Short-Term Memory (LSTM) networks, a type of deep learning model, have been widely adopted for time series forecasting. **LSTMs** excel at identifying long-term dependencies, making them particularly suitable for stock price prediction.

This project aims to predict future opening stock prices using historical market data. The dataset is stored in the *StockDataSP500* folder, containing stock records for multiple companies with various features for analysis. The main objectives include:

- Preprocessing stock data, including data loading, normalization, and conversion into time series data with window sizes of 10, 20, and 50 days.
- Developing and training an **LSTM** model to predict future opening prices using historical data.
- Comparing how different window sizes affect prediction accuracy to evaluate **LSTM**'s effectiveness in stock market forecasting.

This project is a collaborative effort between Hengsheng Li and Hengrongfei Li, with responsibilities divided as follows:
- **Hengsheng Li** handled data processing tasks, including data loading, normalization, window-based sequence creation, and splitting data into training/test sets.
- **Hengrongfei Li** focused on **LSTM** model development and evaluation, including architecture design, model training, hyperparameter tuning, and performance analysis.

Through this research, we aim to apply the knowledge gained from the *Python for Engineering Data Analysis* course to a practical scenario, exploring the effectiveness of **LSTM** networks in stock price prediction.