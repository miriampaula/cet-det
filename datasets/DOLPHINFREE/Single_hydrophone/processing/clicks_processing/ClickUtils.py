# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:12:14 2021

@author: Loïc
"""

#%% Packages importations
import csv
import os

import pandas as pd
import numpy as np

from scipy.signal import butter, filtfilt

#%% Functions process
def import_csv(csv_name, folder, separator, useless=True):
    """
    Parameters
    ----------
    csv_name : STRING
        Name of the csv file that needs to be imported
    folder : STRING, optional
        Path to the folder containing csv file. 
        The default is data_folder (see Parameters section).
    separator : STRING
        Separator for folders.
    useless : BOOLEAN, optional
        Does the file contains useless infos after the table ? 
        If True, remove those infos

    Returns
    -------
    data : LIST
        List containing all rows of the csv file. 
        Exclusion of last lines if they don't end with .wav.
    """
    data = []
    
    # read data csv
    with open(folder + separator + csv_name, newline="") as csv_file:
        lines = csv.reader(csv_file, delimiter=',')
        for row in lines:
            data = data + [row]
    csv_file.close()
    
    # clean csv (last lines contain useless info)
    n = 0
    while useless:
        # in case wrong use of the function
        if n == len(data):
            useless = False
            n = 0
            raise Exception("First column contains no audio file name")
        # exit the loop when audio is found
        if data[-n][0].endswith(".wav"):
            useless = False 
        # search audio in next line
        else:
            n += 1
    
    data = data[0:-n]
    return data

def get_csv(csv_folder, slash="\\"):
    """
    Parameters
    ----------
    csv_folder : STRING
        Path to a folder containing ONLY csv file.
    slash : STRING
        Separator for folders.

    Returns
    -------
    data_20_21 : DATAFRAME
        Contains the inforamtion inside the .CSVs.
    sorted_names : LIST
        Names corresponding to audio files in the first column of the .CSVs.
    """
    csv_names = [a for a in os.listdir(csv_folder) if a.endswith('.csv')]

    # import data
    data = import_csv(csv_names[0], csv_folder, separator=slash)
    for i in range(1,len(csv_names)):
        data = data + import_csv(csv_names[i], csv_folder, separator=slash)[1:]
    data_frame = pd.DataFrame(data=data[:][1:], columns=data[:][0])

    # change dtype for selected columns
    for i in range(3,17):
        data_frame.iloc[:,i] = data_frame.iloc[:,i].astype(int) 

    # fetch audio files names
    audio_names = np.copy([])
    for filename in data_frame["Fichier Audio"]:
        if filename.endswith(".wav"):
            audio_names = np.append(audio_names, filename.replace('/', slash))

    # sort audio files names
    # We want to sort it by year
    years = np.unique([path[4:8] for path in audio_names])
    sorted_names = np.array([], dtype='<U49')
    # and inside each year, sort it by date.
    for year in years:
        idx_to_sort = np.where(np.array([path[4:8] for path in audio_names]) == year)[0]
        temp_path = np.copy(audio_names[idx_to_sort])
        temp_path.sort()
        sorted_names = np.append(temp_path, sorted_names)
            
    return data_frame, sorted_names

def butter_pass_filter(data, cutoff, fs, order=1, mode='high'):
    """
    Parameters
    ----------
    data : NUMPY ARRAY
        Audio signal (1D array)
    cutoff : INT or FLOAT
        Frequency limit for the filter.
    fs : INT
        Sample rate of the audio given as 'data'
    order : INT, optional
        Order of the highpass filter. The default is 1.
    mode : STRING, optional
        Mode of the filter (higpass or low pass). Must be 'high' or 'low'
    Returns
    -------
    y : NUMPY ARRAY
        Filtered signal.
    """
    
    normal_cutoff = cutoff / (fs/2)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype=mode, analog=False)
    y = filtfilt(b, a, data)
    return y

def TeagerKaiser_operator(audio):
    """
    Parameters
    ----------
    audio : NUMPY ARRAY
        Audio signal (1D array).

    Returns
    -------
    tk_signal : NUMPY ARRAY
        Signal energy computed with teager kaiser operator (1D array).
    """
    # Define Teager-Kaiser operator
    tk_signal = np.empty(audio.shape)
    # Formula : TK = x(t)**2 - x(t-1)*x(t+1)
    tk_signal[1:(audio.shape[0]-1)] = (audio[1:(audio.shape[0]-1)]**2) - \
        (audio[0:(audio.shape[0]-2)]*audio[2:(audio.shape[0])]) 
    tk_signal[0] = 0  # set first and last value to 0 (neutral)
    tk_signal[-1] = 0
    
    return tk_signal


    
#%% Functions plots
def get_category(samples, names, data_frame, category='acoustic'):
    """
    Parameters
    ----------
    samples : LIST OF STRINGS.
        Origin of each click,
        should be like ['SCW1807_20200711_083600_extract0.npy']).
    names : LIST OF STRINGS
        Names of each recording,
        (should be like ['11072020/SCW1807_20200711_082400.wav']).
    data_frame : PANDAS DATAFRAME
        Dataframe containing all csv infos.
    category : TYPE, optional
        Category to extratc from dataframe. The default is 'acoustic'.
        Should be 'acoustic', 'fishing_net', 'behavior', 'beacon', 'date',
        'number', 'net' or 'none'
        
    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    use_samples = np.array([file[:23] for file in samples])
    use_names = np.array([name.split('/')[-1][-27:-4] for name in names])
    
    cat_list = np.zeros(use_samples.shape, dtype="object")
    
    if category == 'acoustic':
        start, stop = 3, 10
    
    elif category == 'fishing_net':
        start, stop = 10, 13
    
    elif category == 'behavior':
        start, stop = 13, 16
    
    elif category == 'none':
        return np.array(['none']*len(use_samples))
    
    for i, file in enumerate(use_samples):
        idx_ind = np.where(file == use_names)[0][0]
        if category == 'beacon':
            cat_list[i] = data_frame.iloc[idx_ind, 17]
        elif category == 'date':
            cat_list[i] = data_frame.iloc[idx_ind, 1]
        elif category == 'number':
            cat_list[i] = data_frame.iloc[idx_ind, 18]
        elif category == 'net':
            cat_list[i] = data_frame.iloc[idx_ind,19]
        else: 
            column = np.argmax(data_frame.iloc[idx_ind, start:stop])+start
            cat_list[i] = data_frame.columns[column]
    
    if category == 'beacon':
        NC = np.where(get_category(samples, names, data_frame, category='acoustic')
                      == 'AV')[0]
        cat_list[NC] = 'NC'
    
    return cat_list