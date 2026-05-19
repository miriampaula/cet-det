# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 16:03:01 2022

@author: loic
title: Make useful data from detections
"""

#%% Importations
import os
import numpy as np
import pandas as pd

#%% Parameters 
# Paths
csv_f = "./../CSV_data"     # Path to csv data
res_f = "./Results"         # Path to save results
save_f = "peaks_02052022"   # Folder containing detections
version = "02052022" 

from clicks_processing.ClickUtils import get_csv, get_category
data_20_21, audio_paths = get_csv(csv_f, slash="/")

#%% Main
files_in_folder = os.listdir(os.path.join(res_f, save_f))

# get number of clicks
click_per_file = np.zeros(len(files_in_folder))
for i,file in enumerate(files_in_folder):
    click_per_file[i] = len(np.load(os.path.join(res_f, save_f, file)))

print(f"We detected {int(np.sum(click_per_file))} clicks in the audios")

# save each category associated to a click
if input('Save table with categories [Y/n] ? ') == 'Y':
    data_to_save = pd.DataFrame(click_per_file.astype(int), columns=['number_of_clicks'])
    for cat in ['acoustic', 'fishing_net', 'behavior', 'beacon', 'date', 'number', 'net']:
        data_to_save[cat] = get_category(files_in_folder, audio_paths, data_20_21, cat)

    data_to_save['audio_names'] = [file[:-10] for file in files_in_folder]
    data_to_save.to_csv(os.path.join(res_f, "number_of_clicks_" + version + ".csv"),
        index=False, index_label=False)

