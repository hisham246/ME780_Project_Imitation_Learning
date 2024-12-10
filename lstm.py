import torch
import torch.nn as nn

class PoseToControlLSTM(nn.Module):
    def __init__(self, control_dim, lstm_hidden_size=128, lstm_num_layers=2):
        super(PoseToControlLSTM, self).__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=31,              # Number of input features (odom + laser_scan size)
            hidden_size=lstm_hidden_size,  # Hidden state size
            num_layers=lstm_num_layers,    # Number of LSTM layers
            batch_first=True            # Input is (batch, seq, feature)
        )

        # Fully connected layers after LSTM
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, control_dim)
        )

    def forward(self, odom_input, laser_scan):
        """
        Forward pass of the network.
        odom_input: Tensor of shape (batch, odom_features)
        laser_scan: Tensor of shape (batch, laser_features)
        """

        # Ensure inputs are 3D: (batch, seq_len=1, features)
        if odom_input.dim() == 2:
            odom_input = odom_input.unsqueeze(1)  # Add sequence dimension
        if laser_scan.dim() == 2:
            laser_scan = laser_scan.unsqueeze(1)  # Add sequence dimension

        # Concatenate odom_input and laser_scan along the feature dimension
        total_input = torch.cat((odom_input, laser_scan), dim=2)  # Concatenate on the last dimension

        # Pass through LSTM
        lstm_out, _ = self.lstm(total_input)  # Output is (batch, sequence_length, hidden_size)

        # Take the output from the last time step
        lstm_last_output = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Pass through fully connected layers
        output = self.fc(lstm_last_output)

        return output