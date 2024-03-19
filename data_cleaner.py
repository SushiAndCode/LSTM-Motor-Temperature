import pandas as pd
import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import openpyxl
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt



ACCELERATION_TIME_THRESHOLD = 15

#Lowpass filter
def butter_lowpass_filter(data, cutoff, fs = 20, order=4):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


#Function to check if in a specific interval there is no brake
def no_value_interval(signal, index, offset, max_offset):
    #Calc. of the range's end
    end_range = index + offset
    #Check if the value calculated is admittable, else set it to the end of the dataframe
    if end_range > max_offset:
        end_range = max_offset
    for i in range(index, end_range):
        if signal[i] > 0:
            return False
    #True if only brake was always at 0
    return True


#Plot filtered braking signal with movemean filter
#To use the function in main.py: a.plt_filtered_brake(target_tel, target_tel['Brake'])
def plt_filtered_brake(df, braking):
    #filtered signal with movemean
    filtered_values = filter_movmean(df, column='Brake_Ant', filter_window = 10, lower_lim = 1) #array
    result = np.zeros(len(filtered_values), dtype=float, order='C')
    for i in range(len(filtered_values)):
        if np.array_equal(braking.array[i-4:i+1], np.zeros(5, dtype=float, order='C')) or no_value_interval(braking.array, i, 5, len(filtered_values)) :
            result[i] = 0.0
        else:
            result[i] = filtered_values[i]
    support_df = pd.DataFrame({'Brake_Ant': result })
    df["Brake_Ant"] = support_df["Brake_Ant"]
    return df



#Function that filters the tps based on time delta and signal amplitude
def throttle_filter(data):
    #Set of data with the throttle values
    throttle_values = data[["Throttle_Raw","time"]]
    #I remove everything above 95%
    throttle_values.loc[throttle_values["Throttle_Raw"] > 95, "Throttle_Raw"] = 95
    #Array with the value of the tps
    array_tps_value = throttle_values["Throttle_Raw"].to_numpy()
    #Array with the value of time
    array_time_value = throttle_values["time"].to_numpy()
    #Dictionary with the indexes of the non relevant accelerationss
    index_bad_acceleration = {}
    #Dictionary with the accelerations that surpass a certain threshold
    map_relevant_accelerations = {}
    #Acceleration string to create the key of the dictionary
    accelerazione = "Acc"
    #counter for the accelerations
    n_accelerazioni = 1
    #Counter for bad accelerations
    n_relevant_acceleration = 1
    #Index to go through the array tps value
    index = 0
    flag = 1
    while index < array_tps_value.size:
        #Check if the tps starts
        if (array_tps_value[index] >= 1) and flag == 1:
            #Set flag to 0 to show the acceleration starts
            flag = 0
            #Save the index where the accelerations starts
            index_start_acceleration = index

        #If the accelerations lasts less than a threshold i eliminate the acceleration
        if (array_tps_value[index] < 0.5 ) and flag == 0:
            #index of the end of the acceleration
            index_end_acceleration = index
            #Check the time of the acceleration
            if index_end_acceleration - index_start_acceleration < ACCELERATION_TIME_THRESHOLD:
                #Key creation
                accelerazione_n = accelerazione + str(n_accelerazioni)
                n_accelerazioni = n_accelerazioni + 1
                #Update the dictionart
                index_bad_acceleration[accelerazione_n] = [index_start_acceleration,index_end_acceleration]
            else:
                #Key creation
                accelerazione_relevant = accelerazione + str(n_relevant_acceleration)
                n_relevant_acceleration = n_relevant_acceleration + 1
                #Update the dictionary
                map_relevant_accelerations[accelerazione_relevant] = [index_start_acceleration,index_end_acceleration]

            #Set the flag to one - The acceleration ends
            flag = 1
        #Increment the index to continue the cycle
        index = index + 1

    #Application the filter
    #Iterate though the non relevant accelerationss
    for accelerazione in index_bad_acceleration:
        #Set the value inside the tps array = i eliminate the non relevant spikes
        array_tps_value[index_bad_acceleration[accelerazione][0]:index_bad_acceleration[accelerazione][1]+1] = 0
    
    #If the max of the value of one acceleration is lower than the threshold i eliminate the acceleration
    for accelerazione in map_relevant_accelerations:
            if(max(array_tps_value[map_relevant_accelerations[accelerazione][0]:map_relevant_accelerations[accelerazione][1]]) < 95):
                array_tps_value[map_relevant_accelerations[accelerazione][0]:map_relevant_accelerations[accelerazione][1]+1] = 0
    
    #Update the dataframe with the new values
    df = pd.DataFrame({'Throttle_Raw': array_tps_value })
    df = df.assign(time = array_time_value)

    data["time"] = df["time"]
    data["Throttle_Raw"] = df["Throttle_Raw"]

    return data


#Filtering function, moving average approach
def filter_movmean(df, column, filter_window , lower_lim):
    
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



#PATH LOG SIMONE IMOLA
#path = "C:/Users/Simone/OneDrive/Documenti/Simone/Universita/Moto_polimi/Innovation/python_test_file/preprocessing_NN/csv_imola_NN/turno 1"
#PATH LOG SIMONE MISANO
path = "C:/Users/Simone/OneDrive/Documenti/Simone/Universita/Moto_polimi/Innovation/Inn_Elettrica/python_test_file/preprocessing_NN/clean_csv"

#iterate throught the file of the directory to pick the csv file 
directory_names = os.listdir(path)
list_of_dataframes = []
for file in os.listdir(path):
    if os.path.splitext(file)[1] == ".csv":
        list_of_dataframes.append(pd.read_csv(path + "/" + file))   
#Programmed like this for the future case of concatenating more dataframes
logs = pd.concat(list_of_dataframes)
#Reset the index to make it in cardinal order
logs.reset_index(drop = True, inplace = True)

print("INSERISCI COMANDO:")
print("1- Filtro acceleratore frenata")
print("2- Filtraggio spike e rumore")
print("3- Print")
print("4- Taglio sopra al csv")
print("5- Filtro passa basso")
print("6- Tolgo i valori negativi")
print("7- Solo tagliare i dati")
print("8- Downsampling")
print("9- Filtraggio colonne")
print("10- Normalizzazione")
print("11- Movemean")
print("12- For battery Current")
print("13- Watch Columns")
choice = input()

if choice == "1":
    #Filter of the throttle
    logs = throttle_filter(logs)
    # Set min tps to 0
    logs['Throttle_Raw'] = logs['Throttle_Raw'].apply(lambda x : x if x > 0 else 0)
    # Reset index to start from 0
    logs.reset_index(inplace=True, drop = True)
    #Filter of the brake
    logs = plt_filtered_brake(logs, logs["Brake_Ant"])
    norm_df = logs
    # Export this excel
    norm_df.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "2":
    print("Inserisci il canale da filtrare:")
    channel = input()
    print("Inserire differenza in percentuale")
    percent = float(input())
    #The same for the Motor speed
    for ii in range(len(logs.index)):
        if ii > 0:
            if logs.loc[ii, channel] == 0:
                logs.loc[ii, channel] = logs.loc[ii-1, channel]
            elif abs(((logs.loc[ii-1, channel] - logs.loc[ii, channel])/logs.loc[ii, channel])) > percent:
                logs.loc[ii, channel] = logs.loc[ii-1, channel]
    #reset the index
    logs.reset_index(inplace = True,drop=True)
    logs.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "3":
    print("Inserisci colonna da vedere")
    channel = input()
    plt.plot(logs[channel])
    plt.show()
elif choice == "4": 
    #df = pd.DataFrame(df['Motor_Temperature'])
    logs = logs[4:-4]
    logs.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "5":
    print("Inserisci colonna da filtrare:")
    column = input()
    print("Inserisci frequenza di taglio")
    ft = float(input())
    arr = butter_lowpass_filter(logs[column],ft,20,4)
    arr = np.array(arr)
    df = pd.DataFrame({column: arr })
    logs[column] = df[column]
    logs.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "6":
    print("Inserisci colonna da filtrare:")
    column = input()
    # Set min tps to 0
    logs[column] = logs[column].apply(lambda x : 0 if x < 0 else x)
    logs.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "7":
    print("Inserisci la colonna da controllare:")
    channel = input()
    plt.plot(logs[channel])
    plt.show()
    print("Inserisci Inizio")
    inizio = int(input())
    print("Inserisci Fine")
    fine = int(input())
    #Remove useless data
    logs = logs[inizio:fine]
    #I reset the index
    logs.reset_index(inplace = True, drop = True)
    logs.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "8":
    # Downsample dataframe, 100hz is too much data repeated
    logs = logs[::5]
    #I reset the index
    logs.reset_index(inplace=True)
    logs.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "9":
    #Save the columns I'm interested in
    saved = ["time","Motor_Current","Motor_Voltage","Cell_Tot","Motor_Temperature","Battery_Current","Throttle_Torque","Throttle_Raw","Brake_Ant","Motor_Speed","Temp_Max"]
    logs = logs[saved]
    logs.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "10":
    logs["Temp_Max"]= logs["Temp_Max"]/70
    logs["Motor_Temperature"]= logs["Motor_Temperature"]/160
    logs["Battery_Current"]= logs["Battery_Current"]/500
    logs["Motor_Current"]= logs["Motor_Current"]/600
    logs["Motor_Voltage"]= logs["Motor_Voltage"]/130
    logs["Throttle_Torque"]= logs["Throttle_Torque"]/100
    logs["Throttle_Raw"]= logs["Throttle_Raw"]/100
    logs["Brake_Ant"]= logs["Brake_Ant"]/20
    logs["Motor_Speed"]= logs["Motor_Speed"]/8000
    logs["Cell_Tot"]= logs["Cell_Tot"]/130
    logs.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "11":
    print("Inserisci colonna da filtrare")
    channel = input()
    print("Inserisci window mov mean")
    movmean_window = int(input())
    print("Inserisci low_lim")
    low_lim = float(input())
    #Moving average on the parameters
    motorparameter = filter_movmean(logs, channel,movmean_window,low_lim)
    #Create a support dataframe 
    df = pd.DataFrame({ channel: motorparameter})
    #Update the value of the dataframe after the filtering
    logs[channel] = df[channel]
    #reset the index
    logs.reset_index(inplace = True,drop=True)
    logs.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "12":
    logs['Battery_Current'] = logs['Battery_Current'].apply(lambda x : x if x < 6553.5/2 else None)
    logs.to_csv(path  + "/" + "cleaned_csv.csv")
elif choice == "13":
    print(logs.columns)
    
    


    


    
    




