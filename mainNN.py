import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import lfilter, butter, filtfilt
from torch import nn
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import random
import models 


#Filtering function, moving average approach
def filter_movmean(df, column, filter_window , lower_lim=1):
    
    # THRESHOLDS (optimized for braking)

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

# From tensors input (x) and label class (y) for the model training 
def prepare_input(data: torch.tensor, target_col: int, col: list, avg: tuple, col_avg: list, df: pd.DataFrame,  shuffle=True):

    tgt_avg = 4

    # Preallocate output data
    x = torch.empty(data.size(0)-avg[1]-tgt_avg, len(col)+len(col_avg)*2)
    #y = torch.empty(data.size(0)-avg[1]-tgt_avg)
    #y = torch.tensor(data[avg[1]:-tgt_avg, target_col])
    y = df.loc[avg[1]:, 'Motor_Temperature'].diff()
    y.replace([np.inf, -np.inf], np.nan, inplace=True)
    y = torch.tensor(y.fillna(method='bfill').values, dtype=torch.float32)
    y = y[:-tgt_avg]
    # Index to count from zero the position in the row of the tensor
    r_idx = 0


    # Iterate through all the dataset to create training samples
    for i in range(avg[1], data.size(0)-tgt_avg): 

        # Index to count from zero the position in the column of the tensor
        col_idx = 0

        # Fill with values from original tensor
        for j in col:
            # Pick data in the original tensor based on j indexes from col
            x[r_idx, col_idx] = data[i, j]
            # Move to the next column in the output tensor column counter 
            col_idx += 1
        
        # Fill with 3 averages based on j indexes from col_avg
        for j in col_avg:

            # Short average 
            x[r_idx, col_idx] = torch.mean(data[i-avg[0]:i, j])
            # Move to the next column in the output tensor column counter 
            col_idx += 1
            # Medium average 
            x[r_idx, col_idx] = torch.mean(data[i-avg[1]:i, j])
            # Move to the next column in the output tensor column counter 
            col_idx += 1
            # Long average 
            #x[r_idx, col_idx] = torch.mean(data[i-avg[2]:i, j])
            # Move to the next column in the output tensor column counter 
            #col_idx += 1

        # Setting output label based on index from target_col  
        #y[r_idx] = data[i, target_col]
        #y[r_idx] = data[i-tgt_avg:i+tgt_avg, target_col].sum() / (tgt_avg+1)
        y[r_idx] = y[r_idx-tgt_avg:r_idx+tgt_avg].sum()/(tgt_avg+1)

        # Move to the next row in the output tensor row counter
        r_idx += 1
    
    # Setting the label
    #y = torch.tensor(df[avg[1]:][target_col].pct_change().fillna(method='bfill').values, dtype=torch.float32)
    #y = torch.tensor(pd.DataFrame(data[avg[1]:, target_col]).pct_change().fillna(method='bfill').values)

    # Shuffle if requested
    #if shuffle:

    # Temporary filtering 
    for ii in range(len(x[:,1])):
        if ii > 0:
            if x[ii,1] == 0:
                x[ii,1] = x[ii-1,1]
            elif abs((x[ii,1] - x[ii-1,1])/x[ii,1]) > 0.4:
                x[ii,1] = x[ii-1,1]

    for ii in range(len(x[:,0])):
        if ii == 0:
            if x[ii,0] > 0.06:
                x[ii,0] = 0.01
        if ii > 0:
            if x[ii,0] == 0:
                x[ii,0] = x[ii-1,0]
            elif abs((x[ii,0] - x[ii-1,0])/x[ii,0]) > 0.9:
                x[ii,0] = x[ii-1,0]

    # Normalize filtere columns
    x[:,1] = x[:,1] / max(x[:,1])  
    x[:,0] = x[:,0] / max(x[:,0])

    return x, y


    
def main():

    #   ----- DATA IMPORT -----
    # File paths for the datasets Leo
    #path_train = 'C:/Users/leona/Documents/PMF/21-23/MS1/D/Electric/repo/Innovation23_oldML/Misano4_clean_resampled20.xlsx'
    #path_val   = 'C:/Users/leona/Documents/PMF/21-23/MS1/D/Electric/repo/Innovation23_oldML/Misano3_clean_resampled20.xlsx'
    #path_test  = 'C:/Users/leona/Documents/PMF/21-23/MS1/D/Electric/repo/Innovation23_oldML/Misano1-1_clean_resampled20.xlsx'

    # File paths for the datasets Simo
    path_train = "C:/Users/Simone/OneDrive/Documenti/Simone/Universita/Moto_polimi/Innovation/python_test_file/preprocessing_NN/Cleaned_Log_csv/Misano_Turno4.xlsx"
    path_val = "C:/Users/Simone/OneDrive/Documenti/Simone/Universita/Moto_polimi/Innovation/python_test_file/preprocessing_NN/Cleaned_Log_csv/Misano_Turno3_1.xlsx"
    path_test = "C:/Users/Simone/OneDrive/Documenti/Simone/Universita/Moto_polimi/Innovation/python_test_file/preprocessing_NN/Cleaned_Log_csv/Misano_Turno1.xlsx"

    # Loading csv as tensors
    train_df = pd.read_excel(path_train)#.drop(columns='Unnamed: 0')
    val_df = pd.read_excel(path_val)#.drop(columns='Unnamed: 0')
    test_df = pd.read_excel(path_test)#.drop(columns='Unnamed: 0')

    # Downsample
    train_df = train_df[::10] 
    val_df = val_df[::10] 
    test_df = test_df[::10] 

    train_tensor = torch.tensor(train_df.values)
    val_tensor = torch.tensor(val_df.values)
    test_tensor = torch.tensor(test_df.values)

    # Getting input output pairs for training
    x_train, y_train = prepare_input(train_tensor, 5, [3, 4, 8], (50, 400), [3], train_df)
    x_val, y_val = prepare_input(val_tensor, 5, [3, 4, 8], (50, 400), [3], val_df)
    x_test, y_test = prepare_input(test_tensor, 5, [3, 4, 8], (50, 400), [3], test_df)
    print('\nDATA IMPORTED')

    #   ----- MODEL -----

    # Setting device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nRUNNING ON: {device}')

    # Model parameters
    hidden_size = 128
    batch_size = 512 # Training batch size

    # Set learning rate and number of epochs to train over
    lr = 1e-4
    n_epochs = 500

    # Initialize the model
    model = models.DeepLinNN(n_features=x_train.size(1), hidden_channels=hidden_size).to(device)
    model_total_params = sum(p.numel() for p in model.parameters())
    print(f'\nNumber of parameters of the model: {model_total_params}')

    # Initialize the loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    #   ----- TRAINING LOOP -----
    # Loop over epochs
    t_losses, v_losses = np.empty(0), np.empty(0)
    for epoch in range(n_epochs):
        
        # Train step
        model.train()
        optimizer.zero_grad()
        # move inputs to device
        x = x_train.to(device)
        y  = y_train.to(device)
        # Forward Pass
        preds = model(x).squeeze()
        # Compute loss
        loss = criterion(preds.view(-1, 1), y.view(-1, 1)) 
        train_loss = loss.item()
        loss.backward()
        optimizer.step()
        t_losses = np.append(t_losses, train_loss)

        # Valid step
        model.eval()
        # move inputs to device
        x = x_val.to(device)
        y  = y_val.to(device)
        # Forward Pass
        preds = model(x).squeeze()
        # Compute loss
        loss = criterion(preds, y) 
        val_loss = loss.item()
        v_losses = np.append(v_losses, val_loss)

        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {val_loss}')

    # Generate final predictions
    with torch.no_grad():
        preds_train = model(x_train.to(device)).squeeze().cpu()
        preds_val = model(x_val.to(device)).squeeze().cpu()
        preds_test = model(x_test.to(device)).squeeze().cpu()

    # Plots 
    plt.figure()
    plt.plot(y_train, label='Real train')
    plt.plot(preds_train, label='Pred train')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t_losses/max(t_losses), label='Train_loss')
    plt.plot(v_losses/max(v_losses), label='Validation loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(y_val, label='Real val')
    plt.plot(preds_val, label='Pred val')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(y_test, label='Real test')
    plt.plot(preds_test, label='Pred test')
    plt.legend()
    plt.show()

    '''
    # Setup dataloader for automatic batch management
    train_dataloader = DataLoader(training_data, batch_size=256, shuffle=False)

    # Lists to store training and validation losses
    t_losses, v_losses = [], []
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
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = train_loss #/ len(trainloader)
        t_losses.append(epoch_loss)

        # validation step
        model.eval()
        # Loop over validation dataset
        for x, y in testloader:
            with torch.no_grad():
                x, y = x.to(device), y.squeeze().to(device)
                preds = model(x).squeeze()
                error = criterion(preds, y)
            valid_loss += error.item()
        valid_loss = valid_loss 
        v_losses.append(valid_loss)
            
        print(f'Epoch: {epoch}, Train Loss: {epoch_loss}, Valid Loss: {valid_loss}')

    '''
    


if __name__ == "__main__":
    main()
