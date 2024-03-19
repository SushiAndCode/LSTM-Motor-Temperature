import argparse
import os.path as osp
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score


class DeepLinNN(torch.nn.Module):

    def __init__(self, n_features, hidden_channels):

        super().__init__()

        # Deep Layers
        self.layer1 = torch.nn.Linear(n_features, hidden_channels)
        self.layer2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.layer3 = torch.nn.Linear(hidden_channels, int(round(hidden_channels/2)))
        self.layer4 = torch.nn.Linear(int(round(hidden_channels/2)), 1)

        # Activation functions
        self.leakyrelu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        # Dropout layer
        self.dropout = torch.nn.Dropout(0.2)

        # Normalization layers
        self.bnorm = torch.nn.BatchNorm1d(hidden_channels)
        self.lnorm1 = torch.nn.LayerNorm(hidden_channels)
        self.lnorm2 = torch.nn.LayerNorm(int(round(hidden_channels/2)))

    def forward(self, x):
        hidden1 = self.tanh(self.layer1(x))
        hidden2 = self.dropout(self.leakyrelu(self.lnorm1(self.layer2(hidden1))))
        hidden3 = self.dropout(self.leakyrelu(self.lnorm2(self.layer3(hidden2))))
        y_pred = self.layer4(hidden3)
        
        return y_pred

    '''def evaluate(self, y_pred, edge_label):
        logsoft = torch.nn.LogSoftmax(dim=1)
        y_pred = torch.argmax(logsoft(y_pred), dim=1)
        return f1_score(edge_label.detach().cpu(), y_pred.detach().cpu(), average='micro')'''
    

class LSTM(torch.nn.Module):

  def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, use_cuda=False, dropout=0.2):
    '''
    n_features: number of input features (1 for univariate forecasting)
    n_hidden: number of neurons in each hidden layer
    n_outputs: number of outputs to predict for each training example
    n_deep_layers: number of hidden dense layers after the lstm layer
    sequence_len: number of steps to look back at for prediction
    dropout: float (0 < dropout < 1) dropout ratio between dense layers
    '''
    super().__init__()

    self.n_lstm_layers = n_lstm_layers
    self.nhid = n_hidden
    self.use_cuda = use_cuda # set option for device selection
    self.device = 'cuda' if use_cuda else 'cpu'

    # LSTM Layer
    self.lstm = torch.nn.LSTM(n_features,
                    n_hidden,
                    num_layers=n_lstm_layers,
                    batch_first=True) # As we have transformed our data in this way
    
    # first dense after lstm
    self.fc1 = torch.nn.Linear(n_hidden * sequence_len, n_hidden) 
    # Dropout layer 
    self.dropout = torch.nn.Dropout(p=dropout)

    # Create fully connected layers (n_hidden x n_deep_layers)
    dnn_layers = []
    for i in range(n_deep_layers):
        # Last layer (n_hidden x n_outputs)
        if i == n_deep_layers - 1:
            dnn_layers.append(torch.nn.ReLU())
            dnn_layers.append(torch.nn.Linear(self.nhid, n_outputs))
        # All other layers (n_hidden x n_hidden) with dropout option
        else:
            dnn_layers.append(torch.nn.ReLU())
            dnn_layers.append(torch.nn.Linear(self.nhid, self.nhid))
            if dropout:
                dnn_layers.append(torch.nn.Dropout(p=dropout))
    # compile DNN layers
    self.dnn = torch.nn.Sequential(*dnn_layers)

  def forward(self, x):

    # Initialize hidden state
    hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
    cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)

    # move hidden state to device
    if self.use_cuda:
        hidden_state = hidden_state.to(self.device)
        cell_state = cell_state.to(self.device)
            
    self.hidden = (hidden_state, cell_state)

    # Forward Pass
    x, h = self.lstm(x, self.hidden) # LSTM
    x = self.dropout(x.contiguous().view(x.shape[0], -1)) # Flatten lstm out 
    x = self.fc1(x) # First Dense
    return self.dnn(x) # Pass forward through fully connected DNN.