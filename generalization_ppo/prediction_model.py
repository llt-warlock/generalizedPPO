import torch
import torch.nn as nn

class PreferencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the LSTM model
        :param input_size: The number of expected features in the input x (e.g., size of state representation)
        :param hidden_size: The number of features in the hidden state h of the LSTM
        :param output_size: The size of the preference vector to predict
        :param num_layers: Number of recurrent layers (stacked LSTMs)
        """
        super(PreferencePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the LSTM
        :param x: Input sequence of shape (batch, seq_len, features)
        :return: Predicted preference vector
        """
        lstm_out, _ = self.lstm(x)
        # Take the last output in the sequence as the prediction
        predictions = self.linear(PreferencePredictorlstm_out[:, -1, :])
        return predictions

# Example usage:
# Define model
input_size = 10  # Size of the input state representation
hidden_size = 50  # Size of the hidden layers
output_size = 3  # Size of the output preference vector
num_layers = 1  # One layer of LSTM

model = PreferencePredictor(input_size, hidden_size, output_size, num_layers)

# Example input (batch_size, sequence_length, input_size)
x_dummy = torch.rand(5, 7, input_size)  # Batch of 5, sequence length of 7

# Forward pass
preference_prediction = model(x_dummy)

print("Predicted Preference Vector:", preference_prediction)
