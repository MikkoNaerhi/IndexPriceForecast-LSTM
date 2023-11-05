from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    """ A Long Short-Term Memory (LSTM) model for time series prediction.

    Attributes:
    -----------
        lstm (nn.LSTM): The LSTM layers of the model.
        linear (nn.Linear): A linear layer to output the prediction.

    Methods:
    --------
        forward(x): Defines the forward pass of the model.
    """
    def __init__(
        self,
        input_dim:int, 
        hidden_dim:int,
        output_dim:int=1,
        num_layers:int=2
    ):
        """ Initialize the LSTM model.

        Parameters:
        -----------
            input_dim: The number of input features.
            hidden_dim: The number of hidden units in each LSTM layer.
            output_dim: The number of output features. Defaults to 1.
            num_layers: The number of stacked LSTM layers. Defaults to 2.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """ Defines the forward pass of the LSTM model.

        Parameters:
            x: The input data tensor with shape (batch_size, seq_length, input_dim).

        Returns:
            Tensor: The output data tensor with shape (batch_size, output_dim).
        """
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

def load_data(file_name:str) -> pd.DataFrame:
    """ Load the dataset from a CSV file.

    Parameters:
    -----------
        file_name: Data file name.
    
    Returns:
    --------
        DataFrame: A pandas DataFrame containing the loaded data.

    Raises:
    -------
        FileNotFoundError: If the CSV file is not found at the specified path.
    """
    data_path = f'data/{file_name}'
    try:
        data = pd.read_csv(data_path)
        return data
    except:
        raise FileNotFoundError(f"The data file: {file_name} was not found.")

def prepare_sequences(data:np.ndarray, seq_length:int=10) -> Tuple[np.ndarray, np.ndarray]:
    """ Prepares sequences of a given length from time-series data for the LSTM model.

    Parameters:
    -----------
        data: The time-series data.
        seq_length: The length of the sequences to prepare.

    Returns:
    --------
        (X, y): A tuple containing two numpy arrays for the input and target sequences.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return (np.array(X), np.array(y))

def train_model(
    model:LSTMModel,
    train_loader:DataLoader,
    criterion,
    optimizer,
    epochs:int
) -> None:
    """ Trains the LSTM model with the given data loader, loss criterion, optimizer, and number of epochs.

    Parameters:
    -----------
        model (LSTMModel): The LSTM model to train.
        train_loader (DataLoader): The data loader containing training data.
        criterion: The loss function to use for training.
        optimizer: The optimization algorithm to use for training.
        epochs (int): The number of epochs to train the model for.
    """
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()  # Clear gradients from the previous step
            output = model(batch_X)  # Forward pass
            loss = criterion(output, batch_y)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def evaluate_model(
    model:LSTMModel,
    X_test_tensor:torch.Tensor,
    y_test_tensor:torch.Tensor,
    scaler:MinMaxScaler
) -> np.ndarray:
    """ Evaluates the trained LSTM model on the test data.

    Parameters:
    -----------
        model (LSTMModel): The trained LSTM model.
        X_test_tensor (Tensor): The input test data tensor.
        y_test_tensor (Tensor): The target test data tensor.
        scaler: The scaler used to inverse-transform the predicted values.

    Returns:
    --------
        predictions_np: The predicted values inverse-transformed to their original scale.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients for evaluation
        predictions = model(X_test_tensor)
        # Inverse transform the scaled predictions to original scale
        predictions_np = scaler.inverse_transform(predictions.numpy())
        y_test_np = scaler.inverse_transform(y_test_tensor.numpy())
        # Calculate the mean squared error or any other performance metric
        mse = mean_absolute_error(y_test_np, predictions_np)
        print(f'Test MSE: {mse}')
        return predictions_np
    
def plot_results(
    train_data:pd.DataFrame,
    test_data:pd.DataFrame,
    test_predictions:np.ndarray,
    seq_length:int=10
) -> None:
    """ Plots the training data, test data, model predictions, and future predictions.

    Parameters:
    -----------
        train_data: The training data.
        test_data: The test data.
        test_predictions: The model's predictions on the test data.
        seq_length: The sequence length used in model training and prediction.
    """
    # Calculate the index for the test data (continuing from training data)
    train_len = len(train_data)
    test_len = len(test_data)
    
    # Adjust test data index based on sequence length used during prediction
    test_index = range(train_len + seq_length, train_len + test_len)

    plt.figure(figsize=(15,7))

    # Plot training data
    plt.plot(range(train_len), train_data['Price'], label='Training Data')

    # Plot testing data
    plt.plot(test_index, test_data['Price'][seq_length:], label='Actual Test Data')

    # Plot the predictions for the test data
    plt.plot(test_index, test_predictions, label='Test Predictions', color='orange')

    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('assets/pred_results.png')
    plt.show()

def main(plot:bool=True):
    """ The main execution function for training the LSTM model and plotting results.

    Parameters:
    -----------
        plot: If True, plot the results after training and evaluation.
    """
    file_name = 'nifty_index_price_data.csv'

    data = load_data(file_name)

    # Convert date and filter columns
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data = data[['Date','Close']].rename(columns={'Close':'Price'})

    # Forward-fill to handle missing values if any
    data['Price'] = data['Price'].ffill()

    # Split data into training and testing
    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size], data[train_size:len(data)]

    # Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train['Price'].values.reshape(-1, 1))
    train_scaled = scaler.transform(train['Price'].values.reshape(-1, 1))
    test_scaled = scaler.transform(test['Price'].values.reshape(-1, 1))

    # Prepare sequences
    seq_length = 10
    X_train, y_train = prepare_sequences(train_scaled, seq_length)
    X_test, y_test = prepare_sequences(test_scaled, seq_length)

    # Initialize model, criterion and optimizer
    model = LSTMModel(input_dim=1, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).view(-1, seq_length, 1)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test).view(-1, seq_length, 1)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

    train_model(model, train_loader, criterion, optimizer, epochs=100)
    test_predictions = evaluate_model(model, X_test_tensor, y_test_tensor, scaler)

    if plot:
        plot_results(train_data=train, test_data=test, test_predictions=test_predictions)

if __name__ == "__main__":
    main()