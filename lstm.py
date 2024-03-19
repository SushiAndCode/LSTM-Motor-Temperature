import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import lfilter, butter, filtfilt
from torch import nn
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import random
import itertools


#Filtering function, moving average approach
def filter_movmean(df, column, filter_window , lower_lim=1):
    

    #Filter window   
    #The higher the length, the smoother the curve but it will neglect more data (default 5)
    
    #Lower lim
    #Lower threshold, values below that are set to 0 (default 1)
    
    #Converts pandas data to numpy array
    signal = df[column].to_numpy()
    #Eliminate the noise, all the values lower than one of the size of the noise become 0
    for ii in range(0,len(signal)-1):
        if signal[ii] <= lower_lim:
            signal[ii] = 0

    brake_mov_av = np.convolve(signal, np.ones((filter_window)), mode='same')

    brake_mov_av /= filter_window
    return brake_mov_av

#Lowpass filter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


#   ----- MODEL -----
class LSTMForecaster(torch.nn.Module):

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
    self.lstm = nn.LSTM(n_features,
                        n_hidden,
                        num_layers=n_lstm_layers,
                        batch_first=True) # As we have transformed our data in this way
    
    # first dense after lstm
    self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden) 
    # Dropout layer 
    self.dropout = nn.Dropout(p=dropout)

    # Create fully connected layers (n_hidden x n_deep_layers)
    dnn_layers = []
    for i in range(n_deep_layers):
        # Last layer (n_hidden x n_outputs)
        if i == n_deep_layers - 1:
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Linear(self.nhid, n_outputs))
        # All other layers (n_hidden x n_hidden) with dropout option
        else:
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Linear(self.nhid, self.nhid))
            if dropout:
                dnn_layers.append(nn.Dropout(p=dropout))
    # compile DNN layers
    self.dnn = nn.Sequential(*dnn_layers)

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


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

class SequenceDataset():

    def __init__(self, df):
        self.data = df

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])

    def __len__(self):
        return len(self.data)

def make_predictions_from_dataloader(model, unshuffled_dataloader, device):
    model.eval()
    predictions, actuals = [], []
    for x, y in unshuffled_dataloader:
        # move inputs to device
        x = x.to(device)
        y  = y.squeeze().to(device)
        with torch.no_grad():
            p = model(x)
            predictions.append(p)
            actuals.append(y.squeeze())

    predictions = torch.cat(predictions).cpu().numpy()/100
    actuals = torch.cat(actuals).cpu().numpy()/100
    y0 = 0
    t_series_actuals = [y0]
    t_series_pred = [y0]
    for i in range(1,len(actuals)):
        t_series_actuals.append(t_series_actuals[i-1]+actuals[i])
        t_series_pred.append(t_series_pred[i-1]+predictions[i])

    plt.figure()
    plt.plot(t_series_actuals, label='Real')
    plt.plot(t_series_pred, label='Pred')
    plt.legend()
    plt.show()




    plt.figure()
    plt.plot(actuals, label='Real data')
    plt.plot(predictions, label='Predictions')
    plt.legend()
    plt.show()
    return predictions.squeeze(), actuals


# Defining a function that creates sequences and targets as shown above
def generate_sequences(df: pd.DataFrame, tw: int, pw: int, target_columns, drop_targets=False):
    '''
    df: Pandas DataFrame of the univariate time-series
    tw: Training Window - Integer defining how many steps to look back
    pw: Prediction Window - Integer defining how many steps forward to predict

    returns: dictionary of sequences and targets for all sequences
    '''
    data = dict() # Store results into a dictionary
    L = len(df)

    for i in range(L-tw):
        # Option to drop target from dataframe
        if drop_targets:
            df.drop(target_columns, axis=1, inplace=True)

        # Get current sequence  
        sequence = df[i:i+tw].values
        # Get values right after the current sequence
        target = df[i+tw:i+tw+pw][target_columns].values
        data[i] = {'sequence': sequence, 'target': target}
        
    return data

# Defining a function that creates sequences and targets as shown above
def generate_multivariate_sequences(df: pd.DataFrame, tw: int, pw: int, target_columns,
                                    seq_columns, drop_targets=False, step_seq=2):
    '''
    df: Pandas DataFrame of the multivariate time-series
    tw: Training Window - Integer defining how many steps to look back
    pw: Prediction Window - Integer defining how many steps forward to predict

    returns: dictionary of sequences and targets for all sequences
    '''
    data = dict() # Store results into a dictionary
    L = len(df)
    #y = df[target_columns].pct_change()
    y = df[target_columns].diff()
    y.replace([np.inf, -np.inf], np.nan, inplace=True)
    y = torch.tensor(y.fillna(method='bfill').values, dtype=torch.float32)
    yy = []
    tgt_window = 8
    for i in range(int(tgt_window/2), int(L-tw*step_seq-tgt_window/2)):
        # Option to drop target from dataframe
        if drop_targets:
            df.drop(target_columns, axis=1, inplace=True)

        # Get current sequence  
        sequence = df[i:i+tw*step_seq:step_seq][seq_columns].values
        # Get values right after the current sequence
        #target = df[i+tw:i+tw+pw][target_columns].values
        #target = y[i+tw*step_seq:i+tw*step_seq+pw]
        target = y[i+tw*step_seq]

        # Averaging diff of the sequences
        target = y[int(i+tw*step_seq-tgt_window/2):int(i+tw*step_seq+tgt_window/2)].sum()/(tgt_window+1)*100
        data[int(i-tgt_window/2)] = {'sequence': sequence, 'target': target}
        
        #data[i] = {'sequence': sequence, 'target': target}
        yy.append(target.item())
        
    return data

def filter_current(df):
    # Filter current
    '''    
    for ii in range(len(df['Battery_Current'])):

        if ii == 0:
            if df.loc[ii,'Battery_Current'] > 0.06:
                df.loc[ii,'Battery_Current'] = 0.01
        if ii > 0:
            if df.loc[ii,'Battery_Current'] == 0:
                df.loc[ii,'Battery_Current'] = df.loc[ii-1,'Battery_Current']
            elif abs((df.loc[ii,'Battery_Current'] - df.loc[ii-1,'Battery_Current'])/df.loc[ii,'Battery_Current']) > 0.5:
                df.loc[ii,'Battery_Current'] = df.loc[ii-1,'Battery_Current']
    # Normalize again filtered current
    df['Battery_Current'] = df['Battery_Current']/max(df['Battery_Current'])
    '''
    # Downsample all the dataframe
    df = df[::10] 

    return df

def main():
    path = "C:/Users/leona/Documents/PMF/21-23/MS1/D/Electric/repo/Innovation23_oldML/Misano_Turno1_clean.csv"
    df1 = pd.read_csv(path)
    path_test = "C:/Users/leona/Documents/PMF/21-23/MS1/D/Electric/repo/Innovation23_oldML/Misano_Turno3_1_clean.csv"
    df2 = pd.read_csv(path_test)
    path_test = "C:/Users/leona/Documents/PMF/21-23/MS1/D/Electric/repo/Innovation23_oldML/Misano_Turno4_clean.csv"
    df3 = pd.read_csv(path_test)
    path_test = "C:/Users/leona/Documents/PMF/21-23/MS1/D/Electric/repo/Innovation23_oldML/Misano_Turno2_clean.csv"
    df4 = pd.read_csv(path_test)
    '''    
    z_scores = stats.zscore(df)

    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df[filtered_entries]

    # Filter outliers   DA RIVEDERE
    for ii in range(len(df.index)):
        if ii > 0:
            if df.loc[ii, 'Motor_Temperature'] == 0:
                df.loc[ii, 'Motor_Temperature'] = df.loc[ii-1, 'Motor_Temperature']
            elif ((df.loc[ii, 'Motor_Temperature'] - df.loc[ii-1, 'Motor_Temperature'])/df.loc[ii, 'Motor_Temperature']) > 0.1:
                df.loc[ii, 'Motor_Temperature'] = df.loc[ii-1, 'Motor_Temperature']

    #df['Motor_Temperature'] = butter_lowpass_filter(data=df['Motor_Temperature'], cutoff=)
    df['Motor_Temperature'] = filter_movmean(df, 'Motor_Temperature', filter_window=5)
    # Fit scalers
    scalers = {}
    df = pd.DataFrame(df.loc[:49000,'Motor_Temperature'])
    df = df[1:]
    for x in df.columns:
        scalers[x] = MinMaxScaler().fit(df[x].values.reshape(-1, 1))

    # Transform data via scalers
    norm_df = df.copy()
    for i, key in enumerate(scalers.keys()):
        norm = scalers[key].transform(norm_df.iloc[:, i].values.reshape(-1, 1))
        norm_df.iloc[:, i] = norm
    '''
    # FIltering currents
    df1 = filter_current(df1)
    df2 = filter_current(df2)
    df3 = filter_current(df3)

    # Downsample test set
    df4 = df4[2000:6000]
    df4 = filter_current(df4)

    nhid = 32 # Number of nodes in the hidden layer
    n_dnn_layers = 2 # Number of hidden fully connected layers
    nout = 1 # Prediction Window
    sequence_len = 8 # Training Window

    # Number of features (since this is a univariate timeseries we'll set
    # this to 1 -- multivariate analysis is coming in the future)
    ninp = 3

    # Device selection (CPU | GPU)
    USE_CUDA = torch.cuda.is_available()
    device = 'cuda' if USE_CUDA else 'cpu'
    print(f'\nRunning on: {device}')

    # Initialize the model
    model = LSTMForecaster(ninp, nhid, nout, sequence_len, n_deep_layers=n_dnn_layers, use_cuda=USE_CUDA).to(device)
    model_total_params = sum(p.numel() for p in model.parameters())
    print(f'\nNumber of parameters of the model: {model_total_params}')

    # Set learning rate and number of epochs to train over
    lr = 5e-5
    n_epochs = 500

    # Here we are defining properties for our model

    BATCH_SIZE = 64 # Training batch size
    split = 0.9 # Train/Test Split ratio

    #df4 = df1.loc[4000:5000,:]
    #df1 = df1.loc[5000+1:,:]
    # 'Motor_Temperature', ['Motor_Current', 'Motor_Speed', 'Throttle_Raw']
    # 'Temp_Max', ['Battery_Current', 'Cell_Tot', 'Motor_Speed', 'Throttle_Raw']
    #sequences = generate_sequences(norm_df.Motor_Temperature.to_frame(), sequence_len, nout, 'Motor_Temperature')
    sequences1 = generate_multivariate_sequences(df1, sequence_len, nout, 'Temp_Max', ['Battery_Current', 'Motor_Speed', 'Throttle_Raw'])
    #ds_train = SequenceDataset(sequences1)
    sequences2 = generate_multivariate_sequences(df2, sequence_len, nout, 'Temp_Max', ['Battery_Current', 'Motor_Speed', 'Throttle_Raw'])
    #ds_test = SequenceDataset(sequences2)
    sequences3 = generate_multivariate_sequences(df3, sequence_len, nout, 'Temp_Max', ['Battery_Current', 'Motor_Speed', 'Throttle_Raw'])
    #ds_test = SequenceDataset(sequences2)

    sequences4 = generate_multivariate_sequences(df4, sequence_len, nout, 'Temp_Max', ['Battery_Current', 'Motor_Speed', 'Throttle_Raw'])
    ds_test = SequenceDataset(sequences4)


    # New sequence2 with keys starting from len(sequences1) to merge dictionaries
  
    n1 = len(sequences1)
    n2 = len(sequences2)
    new_keys2 = list(range(n1, n1+n2))
    new_sequences2 = dict.fromkeys(new_keys2)
    count = 0
    for key in sequences2.keys():
        new_sequences2[new_keys2[count]] = sequences2[key]
        count += 1

    sequences1.update(new_sequences2)


    n1 = len(sequences1)
    n3 = len(sequences3)
    new_keys3 = list(range(n1, n1+n3))
    new_sequences3 = dict.fromkeys(new_keys3)
    count = 0
    for key in sequences3.keys():
        new_sequences3[new_keys3[count]] = sequences3[key]
        count += 1

    sequences1.update(new_sequences3)
    ds_train = SequenceDataset(sequences1)


    # Split the data according to our split ratio and load each subset into a
    # separate DataLoader object
    train_len = int(len(ds_train)*split)
    lens = [train_len, len(ds_train)-train_len]
    train_ds, val_ds = torch.utils.data.random_split(ds_train, lens)
    unshuffled_data = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # Initialize the loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)

    '''
    # multivariate data preparation
    # define input sequence
    in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    # horizontally stack columns
    dataset = np.hstack((in_seq1, in_seq2, out_seq))
    print(dataset)
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    print(X.shape, y.shape)'''

    # Lists to store training and validation losses
    t_losses, v_losses = np.empty(0), np.empty(0)
    # Loop over epochs
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0.0, 0.0

        # train step
        model.train()
        # Loop over train dataset
        for x, y in trainloader:
            optimizer.zero_grad()
            # move inputs to device
            x = x.to(device)
            y  = y.squeeze().to(device)
            # Forward Pass
            preds = model(x).squeeze()
            loss = criterion(preds, y) # compute batch loss
            # mean cube error
            #loss = sum([(y[i].detach().cpu()-preds[i].detach().cpu())**3 for i in range(len(y))])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = train_loss #/ len(trainloader)
        t_losses = np.append(t_losses, train_loss)

        # validation step
        model.eval()
        # Loop over validation dataset
        for x, y in testloader:
            with torch.no_grad():
                x, y = x.to(device), y.squeeze().to(device)
                preds = model(x).squeeze()
                error = criterion(preds, y)
            valid_loss += error.item()
        val_loss = valid_loss 
        v_losses = np.append(v_losses, val_loss)
            
        print(f'Epoch: {epoch}, Train Loss: {epoch_loss}, Valid Loss: {valid_loss}')
    

    #plot_losses(t_losses, v_losses)
    make_predictions_from_dataloader(model, trainloader, device)
    make_predictions_from_dataloader(model, unshuffled_data, device)
    #make_predictions_from_dataloader(model, unshuffled_data, device)

    plt.figure()
    plt.plot(t_losses/max(t_losses), label='Train_loss')
    plt.plot(v_losses/max(v_losses), label='Validation loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
