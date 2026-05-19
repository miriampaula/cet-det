# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:05:58 2021

@author: Loïc
title: projection of detected clicks
"""

#%% Packages importations
print("\rImportation of packages...", end="\r")
import os
import numpy as np
import pandas as pd
from librosa import load, stft
from librosa.feature import spectral_centroid
from sklearn.preprocessing import StandardScaler
import umap
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from datetime import datetime
print("Importation of packages done!")

#%% Parameters
print("\rSetting up parameters...", end="\r")
audio_f = "./../Audio_data"                             # Path to recordings 
csv_f = "./../CSV_data"                                 # Path to csv data
res_f = "./Results"                                     # Path to save results
peaks_f = "peaks_02052022"                              # Path to save results
save_features = "features_projections_02052022"
version = "02052022"

# For functions
sr = 512000             # sample rate of the recordings
cut_low = 50000         # frequency cut in highpass
num_order = 1           # order of the highpass filter
n_fft = 256             # size of the fft window
hop_length = n_fft//2   # % overlap of fft windows
print("Parameters ready to use!")


#%% Data
print("\rImportation of csv data", end="\r")
from clicks_processing.ClickUtils import get_csv, butter_pass_filter, get_category
data_20_21, audio_paths = get_csv(csv_f, slash="/")
print("Importation of csv data complete!")


#%%## Extract features for each click #####
# mean freq, median freq, freq std
if input("(Re)compute features? [Y/n] ") == "Y":
    features = np.zeros((0,4))
    linked_files = np.zeros(0, dtype=int)

    print("\nPre execution: Looking for clicks in recordings.")
    for file in range(len(audio_paths)):   #np.array([152])
        # Load and transform audio
        print(f"\r\tLoading features: file {file+1} on {len(audio_paths)}", end='\r')

        signal = load(os.path.join(audio_f, audio_paths[file][4:8], audio_paths[file]),
            sr=None)[0]
        signal_high = butter_pass_filter(signal, cut_low, sr, 
                                         num_order, mode='high')  
        Amplitude_audio = stft(signal_high, n_fft=n_fft, 
            hop_length=hop_length, window='hamming')

        # load clicks positions
        clicks_pos = np.load(os.path.join(res_f, peaks_f, audio_paths[file][-27:-4] + "_peaks.npy"))

        # for each click, extract features
        clicks_pos_spec = np.round(clicks_pos/hop_length).astype(int)
        mean_freq = spectral_centroid(signal_high, sr=sr, n_fft=n_fft, 
            hop_length=hop_length, window='hamming')[0,clicks_pos_spec]
        median_freq = np.argsort(np.abs(Amplitude_audio[:,clicks_pos_spec]), axis=0)[(hop_length+1)//2,:]*sr/256
        freq_std = np.std(Amplitude_audio[:,clicks_pos_spec], axis=0)

        # expend results
        features = np.append(features, np.array([mean_freq, median_freq, freq_std, clicks_pos]).T, axis=0)
        linked_files = np.append(linked_files, np.repeat(file, len(clicks_pos_spec)))

    # save results
    np.save(os.path.join(res_f, save_features, "features.npy"),
        features)
    np.save(os.path.join(res_f, save_features, "linked_files.npy"),
        linked_files)
    print("\nPre execution: Finished.")


### Make UMAP projection ###
print("\nMain execution: Projection.")
features = np.load(os.path.join(res_f, save_features, "features.npy"))[:,:3]
print(features.shape)
linked_files = np.load(os.path.join(res_f, save_features, "linked_files.npy"))
print(f"\tClassification of {len(linked_files)} clicks")

# fit UMAP
if input("Fit UMAP projection ? [Y/n] ") == "Y":
    reducer = umap.UMAP(n_neighbors=150, min_dist=0, random_state=None, low_memory=True, verbose=True)
    scaler = StandardScaler()
    features_nrm = scaler.fit_transform(features)

    reducer.fit(features_nrm[::10])
    embedding1 = reducer.transform(features_nrm)
    res_umap1 = pd.DataFrame({'UMAP1': embedding1[:,0]})
    for i in range(1, 2):
        res_umap1['UMAP'+str(i+1)] = embedding1[:,i]
    res_umap1.to_csv(os.path.join(res_f, save_features, "projection.csv"))
    f_name = os.path.join(res_f, save_features, "reducer.sav")
    pickle.dump(reducer, open(f_name, 'wb'))
else:
    res_umap1 = pd.read_csv(os.path.join(res_f, save_features, "projection.csv"))
print("\tUMAP ready for display!")

use_files = np.copy(linked_files).astype(object)
for file in np.unique(use_files):
    use_files[use_files == file] = audio_paths[file][-27:]

# display UMAP
shuffle=np.random.permutation(res_umap1.shape[0])
results=res_umap1.iloc[shuffle].copy()
for cat in ["date"]: #['acoustic','behavior','fishing_net','net','date']:
    category_list = get_category(use_files, audio_paths, data_20_21, category=cat)[shuffle]
    plt.figure()
    sns.scatterplot(data=results, x='UMAP1', y='UMAP2',
        hue=category_list).set_title(f'UMAP projection of {res_umap1.shape[0]} clicks')
    sns.set_theme(style="ticks", font_scale=2)
    if cat=="date":
        handles, labels = plt.gca().get_legend_handles_labels()
        dates = [datetime.strptime(ts, "%d/%m/%Y") for ts in labels]
        order = np.argsort(dates)
        plt.legend([handles[idx] for idx in order],
                   [labels[idx] for idx in order],
                   loc='lower right')
    else:
        plt.legend(loc='lower right')
    plt.show(block=True)

print("\nMain execution: The End.")


##### MANUAL ZONE: selection of groups ######
print("\nSelection of groups")
# import data
coords = pd.read_csv(os.path.join(res_f, save_features, "projection.csv")).iloc[:,1:]  

# 2 colors : green and red
# parameters found empirically
infos = np.array([
         [14.3, 4, 2, 5.5, 0, 'green'],
         [14.25, 9, 1, 5, 0, 'red']
        ])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

rad_cc = np.zeros( (coords.shape[0],(len(infos))) )
colors_array = np.array(['black'] * coords.shape[0])

for i in range(len(infos)):
    ellipse = Ellipse(xy=(float(infos[i][0]),float(infos[i][1])), 
                      width=float(infos[i][2]),
                      height=float(infos[i][3]),
                      angle=float(infos[i][4]),
                      edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse)
    cos_angle = np.cos(np.radians(180.-float(infos[i][4])))
    sin_angle = np.sin(np.radians(180.-float(infos[i][4])))
    xc = np.array(coords.iloc[:,0]) - float(infos[i][0])
    yc = np.array(coords.iloc[:,1]) - float(infos[i][1])
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 
    rad_cc[:,i] = (xct**2/(float(infos[i][2])/2.)**2) + (yct**2/(float(infos[i][3])/2.)**2)
    colors_array[np.where(rad_cc[:,i] <= 1.)[0]] = infos[i][-1]

ax.scatter(coords.iloc[:,0],coords.iloc[:,1],
           c=colors_array,linewidths=0.3)
plt.show(block=True)

# get idx of the points in each ellipse
green_group = np.where(rad_cc[:,0] <= 1.)[0]
red_group = np.where(rad_cc[:,1] <= 1.)[0]
black_group = np.delete(np.arange(len(rad_cc)), np.unique(np.append(green_group, red_group)))

# # let's see the small groups on spectrograms (comment if data not available)
# green_names = np.copy(linked_files[green_group])
# red_names = np.copy(linked_files[red_group])
# names = np.intersect1d(np.unique(red_names), np.unique(green_names))

# # selecting a random file
# file = names[3]
# signal, sr = load(os.path.join(audio_f, audio_paths[file][4:8], audio_paths[file]),
#     sr=None)
# peaks = np.load(os.path.join(res_f, peaks_f, audio_paths[file][-27:-4] + "_peaks.npy"))
# limit_point = np.argwhere(linked_files == file)[:,0]
# rad_file = rad_cc[limit_point]

# colors = np.append(infos[:,-1], ["black"])
# values = np.repeat(-1, len(rad_file))
# values[np.where(rad_file[:,0] <= 1.)[0]] = 0
# values[np.where(rad_file[:,1] <= 1.)[0]] = 1

# fig, axs = plt.subplots(nrows=2, sharex=True)
# axs[0].specgram(signal, xextent=(0,int(60*sr)))
# for value in np.unique(values):
#     arr = np.zeros(int(60*sr))
#     arr[peaks[values==value]] = 1
#     axs[1].plot(arr, colors[value])
# plt.show(block=True)
print("\tBlack points are echolocation clicks, green and red points are SONARs")


### UMAP projection without SONARs ###
print("\tRe-projection without red and green points")
reduced_features = features[black_group]
reduced_linked_files =  linked_files[black_group]
print(f"\tClassification of {len(reduced_linked_files)} clicks")

# Fit new UMAP
if input("Fit UMAP projection ? [Y/n] ") == "Y":
    reducer = umap.UMAP(n_neighbors=150, min_dist=0.5, n_components=2, random_state=None)
    scaler = StandardScaler()
    reduced_features_nrm = scaler.fit_transform(reduced_features)

    embedding1 = reducer.fit_transform(reduced_features_nrm)
    res_umap = pd.DataFrame({'UMAP1': embedding1[:,0]})
    for i in range(1, 2):
        res_umap['UMAP'+str(i+1)] = embedding1[:,i]
    res_umap.to_csv(os.path.join(res_f, save_features, "projection_without_sonar.csv"))
else:
    res_umap = pd.read_csv(os.path.join(res_f, save_features, "projection_without_sonar.csv"))

# display projection
use_files = np.copy(reduced_linked_files).astype(object)
for file in np.unique(use_files):
    use_files[use_files == file] = audio_paths[file][-27:]

shuffle=np.random.permutation(res_umap.shape[0])
results=res_umap.iloc[shuffle].copy()
for cat in ["date"]: #['acoustic','behavior','fishing_net','net','date']:
    category_list = get_category(use_files, audio_paths, data_20_21, category=cat)[shuffle]
    plt.figure()
    sns.scatterplot(data=results, x='UMAP1', y='UMAP2',
        hue=category_list).set_title(f'UMAP projection of {res_umap.shape[0]} clicks')
    sns.set_theme(style="ticks", font_scale=2)
    if cat=="date":
        handles, labels = plt.gca().get_legend_handles_labels()
        dates = [datetime.strptime(ts, "%d/%m/%Y") for ts in labels]
        order = np.argsort(dates)
        plt.legend([handles[idx] for idx in order],
                   [labels[idx] for idx in order],
                   loc='lower right')
    else:
        plt.legend(loc='lower right')
    plt.show(block=False)
print("End of the analysis.")


### update results (exclude SONARs) ###
if input("Update count of clicks and save new groups ? [Y/n]") == "Y":
    # => mainly anthropogenic clicks, we exclude them and see you next script !
    np.save(os.path.join(res_f, save_features, "idx_clicks_not_from_humans.npy"),
        black_group)
    # Find all previous positions
    features = np.load(os.path.join(res_f, save_features, "features.npy"))[:,-1]
    black_positions = features[black_group]
    black_files = linked_files[black_group]
    for file in np.unique(black_files):
        np.save(os.path.join(res_f, 
                peaks_f+"_without_SONARS", 
                audio_paths[file][-27:-4]+"_cleanpeaks.npy"),
            black_positions[np.where(black_files==file)[0]])

    ##### update count of clicks #####
    curr_numbers = pd.read_csv(os.path.join(res_f, "number_of_clicks_" + version + ".csv"))
    new_count = np.zeros(len(audio_paths))
    for file in range(len(audio_paths)):
        here = np.where(curr_numbers['audio_names'] == audio_paths[file][-27:-4])[0]
        curr_numbers.iloc[here, 0] = np.sum(reduced_linked_files==file)
    curr_numbers.to_csv(os.path.join(res_f, "projection_updated_number_of_clicks_" + "please" + ".csv"),
        index=False)
print("\nEnd of the script.")