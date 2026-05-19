# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:05:58 2021

@author: Loïc
title: detection of clicks in audios
"""

# working directory should be set to "Clicks" folder

#%% Packages importations
print("\rImportation of packages...", end="\r")
import librosa
import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
print("Importation of packages done!")

#%% Parameters
# Paths
print("\rSetting up parameters...", end="\r")

from pathlib import Path
import os

# base = repo root = folder that contains Single_hydrophone and Tetra
base = Path(__file__).resolve().parent.parent

audio_f = base / "Single_hydrophone" / "Audio_data"
csv_f   = base / "Single_hydrophone" / "Visual_observation"
save_f  = base / "Single_hydrophone" / "Audio_data" / "Results" / "peaks_02052022"

os.makedirs(save_f, exist_ok=True)

# Audio parameters
sr = 512000                 # sample rate of the recordings
cut_low = 50000             # frequency cut in highpass
num_order = 1               # order of the highpass filter
distance = int(sr*0.101)    # 0.101 sec between two clicks from the beacon
tolerance = int(sr*0.001)   # tolerance for beacon regularity

# User chosen parameters
sound_thresh = 0.001            # threshold for noise detection
click_size = int(sr*0.0002)     # mean size of a click (observations)
max_length = 500                # maximum length of a click (observations)
mini_space = int(click_size*10) # minimal space between two clicks (observations)
print("Parameters ready to use!")


#%% Data importations
print("\rImportation of csv data", end="\r")
from clicks_processing.ClickUtils import get_csv, butter_pass_filter, TeagerKaiser_operator
data_20_21, audio_paths = get_csv(csv_f, slash="/") 
print("Importation of csv data complete!")


#%%## Detect clics in environnement #####
print("\nMain execution: Looking for clicks in recordings.")
for file in range(len(audio_paths)):
    # Load audio
    signal = librosa.load(os.path.join(str(audio_f), audio_paths[file]), sr=None)[0]


    # transform audio
    signal_high = butter_pass_filter(signal, cut_low, sr, 
                                     num_order, mode='high')  
    tk_signal = TeagerKaiser_operator(signal_high)

    # detection of peaks
    signal_peaks = find_peaks(tk_signal, distance=mini_space, width=[0,max_length], 
        prominence=sound_thresh)[0]
    map_peaks = np.zeros(60*sr, dtype=int)
    map_peaks[signal_peaks] = 1

    # clicks from beacon
    idx_in_data = np.where(data_20_21['Fichier Audio'] == \
                           audio_paths[file].replace('\\','/'))
    signals_data = data_20_21.iloc[idx_in_data[0][0],:]
    cat_acoustic = signals_data.iloc[3:10].astype(int).idxmax(axis=0)

    if ('D' in cat_acoustic):
        # exclude them
        map_clean = np.copy(map_peaks)
        for peak in signal_peaks:
            coord1_low = peak-distance-tolerance
            coord1_up = peak-distance+tolerance
            coord2_low = peak+distance-tolerance
            coord2_up = peak+distance+tolerance
            if (1 in map_peaks[coord1_low:coord1_up]) or (1 in map_peaks[coord2_low:coord2_up]):
                map_clean[peak] = 0
    else:
        map_clean=np.copy(map_peaks)


    # save detections
    np.save(os.path.join(save_f, audio_paths[file][-27:-4] + "_peaks"), 
         np.nonzero(map_clean)[0])
    
    print(f"\r\t1-  {file+1} on {len(audio_paths)}: Found {len(np.nonzero(map_clean)[0])} \
clicks", end='\t\r')
print("\nDetection of clicks in recordings complete!")