import torch
import torch.nn as nn                   # All neural network models, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, h):
        batch_size, seq_len, _ = x.size()
        out, h = self.lstm(x, h)
        out = out[:, -1, :]
        out = self.fc(out)

        return out, h

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hidden = (hidden_state, cell_state)  # HIDDEN is defined as a TUPLE
        return hidden

