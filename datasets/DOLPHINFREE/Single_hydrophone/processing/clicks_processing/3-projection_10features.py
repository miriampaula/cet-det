# -*- coding: utf-8 -*-
"""
Created on Tuesday Jul 05 13:34:38 2021

@author: Loïc
title: Projection of clicks (the complicated one)
"""

#%% Packages importations
print("Importation of packages...")
import os
import umap
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
print("Importation of packages done!")

#%% Parameters
print("\nSetting up parameters...")

csv_f = "./../CSV_data"
features_f = "./calcul_features_octave/clicks_features_octave"
res_f = "./Results"
peaks_f = "peaks_02052022"
save_f = "after_octave"

# For functions
features_name = ["audio_name",
                "frequencepeak",
                 "frequencecentr",
                 "DeltaF_3dB",
                 "Deltat_10dB",
                 "DeltaF_10dB",
                 "Deltat_20dB",
                 "DeltaF_rms",
                 "Deltat_rms",
                 "ICI",
                 "SNR",
                 "sample_pos",
                 "acoustic",
                 "fishing_net",
                 "behavior",
                 "date"]
sr = 512000     # sample rate of recordings
len_audio = 60  # length of recordings (in sec)
rd_state = 42   # seed for randomness

print("Parameters ready to use!")
np.random.seed(rd_state)
from clicks_processing.ClickUtils import get_csv, get_category

#%% Data
print("\nImportation of csv data")
data_20_21, audio_paths = get_csv(csv_f, slash="/")

# associate each click to its categories
if input("\tFetch data and features for each click [Y/n]? ") == "Y":
    list_csv = os.listdir(features_f)
    list_csv.sort()
    list_csv = [csv_ for csv_ in list_csv if csv_.endswith('.csv')]
    all_features = pd.DataFrame(data=None, columns=features_name)

    for file in tqdm(list_csv):
        if file.endswith(".csv"):
            data_csv = pd.read_csv(os.path.join(features_f, file),
                header=None, names=features_name[1:])
            # add more informations
            data_csv["audio_name"] = file[9:-4]
            data_csv["acoustic"] = get_category([file[9:]], data_20_21["Fichier Audio"], data_20_21, "acoustic")[0]
            data_csv["fishing_net"] = get_category([file[9:]], data_20_21["Fichier Audio"], data_20_21, "fishing_net")[0]
            data_csv["behavior"] = get_category([file[9:]], data_20_21["Fichier Audio"], data_20_21, "behavior")[0]
            data_csv["date"] = get_category([file[9:]], data_20_21["Fichier Audio"], data_20_21, "date")[0]

            position_clicks = np.load(os.path.join(res_f, peaks_f, file[9:-4] + "_peaks.npy"))        
            if len(position_clicks) != len(data_csv):
                position_clicks = position_clicks[position_clicks>500]
                position_clicks = position_clicks[position_clicks<(sr*len_audio-500)]

            data_csv["sample_pos"] = position_clicks
            all_features = pd.read_csv(os.path.join(res_f, save_f, "all_features.csv"))
            all_features = pd.concat([all_features, data_csv])

    all_features.to_csv(os.path.join(res_f, save_f, "all_features.csv"), index=False)

else:
    all_features = pd.read_csv(os.path.join(res_f, save_f, "all_features.csv"))
print("Importation of csv data complete!")


#%% UMAP Projection
print("\nUmap projection:")
if input("\tCompute UMAP projection [Y/n]? ") == "Y":
    use_feature = all_features[["frequencepeak",
                                "frequencecentr",
                                 "DeltaF_3dB",
                                 "Deltat_10dB",
                                 "DeltaF_10dB",
                                 "Deltat_20dB",
                                 "DeltaF_rms",
                                 "Deltat_rms",
                                 "ICI",
                                 "SNR"]]

    reducer = umap.UMAP(n_neighbors=150, min_dist=0, 
        random_state=42, low_memory=True, verbose=True)
    reducer.fit(use_feature[::10])
    embedding = reducer.transform(use_feature)

    res_umap = pd.DataFrame({'UMAP1': embedding[:,0]})
    for i in range(1, 2):
        res_umap['UMAP'+str(i+1)] = embedding[:,i]

    res_umap.to_csv(os.path.join(res_f, save_f, "all_features_projection-" +\
        datetime.now().strftime("%d%m%Y") + ".csv"), index=False)

    pickle.dump(reducer, open(os.path.join(res_f, "reducer-" +\
        datetime.now().strftime("%d%m%Y") + ".sav"), 'wb'))

else:
    res_umap = pd.read_csv(os.path.join(res_f, save_f, "all_features_projection-07072022.csv"),
        index_col=False)  

print("Umap complete and saved!")

#%% UMAP display
shuffle = np.random.permutation(res_umap.shape[0])
results = res_umap.iloc[shuffle]
feats = all_features.iloc[shuffle]

for cat in ["date"]:#,"acoustic","behavior","fishing_net"]:
    plt.figure()
    sns.scatterplot(data=results, x='UMAP1', y="UMAP2", hue=feats[cat]).set_title(
        f'UMAP projection of {res_umap.shape[0]} clicks')
    sns.set_theme(style="ticks", font_scale=1.5)

    if cat=="date":
        handles, labels = plt.gca().get_legend_handles_labels()
        dates = [datetime.strptime(ts, "%d/%m/%Y") for ts in labels]
        order = np.argsort(dates)
        plt.legend([handles[idx] for idx in order],
                   [labels[idx] for idx in order],
                   loc='best')
    else:
        plt.legend(loc='best')
    plt.show(block=True)

print("Displays are ready.")


#%% Ok, let's focus on the two groups at the bottom right of the projection 
print("\nFind intruders (not clicks)")
# create an ellipse around each group
infos = np.array([
         [13, .5, 5.1, 10, 44, 'green'],
         [11.3, -4, 1, 4, -20, 'red']
        ])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

rad_cc = np.zeros((results.shape[0],(len(infos))))
colors_array = np.array(['black'] * results.shape[0])

for i in range(len(infos)):
    ellipse = Ellipse(xy=(float(infos[i][0]),float(infos[i][1])), 
                      width=float(infos[i][2]),
                      height=float(infos[i][3]),
                      angle=float(infos[i][4]),
                      edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse)
    cos_angle = np.cos(np.radians(180.-float(infos[i][4])))
    sin_angle = np.sin(np.radians(180.-float(infos[i][4])))
    xc = np.array(results.iloc[:,0]) - float(infos[i][0])
    yc = np.array(results.iloc[:,1]) - float(infos[i][1])
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 
    rad_cc[:,i] = (xct**2/(float(infos[i][2])/2.)**2) + (yct**2/(float(infos[i][3])/2.)**2)
    colors_array[np.where(rad_cc[:,i] <= 1.)[0]] = infos[i][-1]

ax.scatter(results.iloc[:,0],results.iloc[:,1],
           c=colors_array,linewidths=0.3)
plt.show(block=True)

# each group contains the following points
green_group = np.where(rad_cc[:,0] <= 1.)[0]
red_group = np.where(rad_cc[:,1] <= 1.)[0]
black_group = np.delete(np.arange(len(rad_cc)), np.unique(np.append(green_group, red_group)))

# select a random green click and a random red click
green_idx = np.random.choice(green_group)
red_idx = np.random.choice(red_group)

# Corresponding files and positions
green_name, green_pos = feats["audio_name"].iloc[green_idx], feats["sample_pos"].iloc[green_idx]
red_name, red_pos = feats["audio_name"].iloc[red_idx], feats["sample_pos"].iloc[red_idx]
print(f"\tCheck file {green_name} in position {green_pos} (in samples)")
print(f"\tAlso check file {red_name} in position {red_pos} (in samples)")
print("\tThese are clicks from a SONAR (and echoes of SONARS sometimes)")

# Manual checking all_features of files with audacity.
# => These clicks correspond to a SONAR and its echoes.

# Discard these two groups and do a new UMAP.
if input("Save new selection ? [Y/n] ") == "Y":
    all_features.iloc[shuffle].iloc[black_group].to_csv(os.path.join(res_f, save_f, "nless_all_features-" +\
            datetime.now().strftime("%d%m%Y") + ".csv"), index=False)

nless_all_features = pd.read_csv(os.path.join(res_f, save_f, "nless_all_features-07072022.csv"),
    index_col=False)

print("FYI:")
data = [
            ["Clicks", 
                np.mean(feats["frequencepeak"].iloc[black_group]),
                np.std(feats["frequencepeak"].iloc[black_group]),
                np.mean(feats["frequencecentr"].iloc[black_group]),
                np.std(feats["frequencecentr"].iloc[black_group]),
                np.mean(feats["ICI"].iloc[black_group]),
                np.std(feats["ICI"].iloc[black_group])
            ],
            ["Sonars", 
                np.mean(feats["frequencepeak"].iloc[np.append(red_group,green_group)]),
                np.std(feats["frequencepeak"].iloc[np.append(red_group,green_group)]),
                np.mean(feats["frequencecentr"].iloc[np.append(red_group,green_group)]),
                np.std(feats["frequencecentr"].iloc[np.append(red_group,green_group)]),
                np.mean(feats["ICI"].iloc[np.append(red_group,green_group)]),
                np.std(feats["ICI"].iloc[np.append(red_group,green_group)])
            ]
        ]
print(tabulate(data, headers=["Mean frequency peak",
                                "Std frequency peak", 
                                "Mean centroid",
                                "Std centroid", 
                                "Mean ICI",
                                "Std ICI"]))
print("Excluded intruders")


#%% New Projection
print("\nRun new projection")
if input("\tCompute UMAP projection (Bis) [Y/n]? ") == "Y":
    nless_use_feature = nless_all_features[["frequencepeak",
                                "frequencecentr",
                                 "DeltaF_3dB",
                                 "Deltat_10dB",
                                 "DeltaF_10dB",
                                 "Deltat_20dB",
                                 "DeltaF_rms",
                                 "Deltat_rms",
                                 "ICI",
                                 "SNR"]]

    nless_reducer = umap.UMAP(n_neighbors=50, min_dist=1, 
        random_state=rd_state, low_memory=True, verbose=True)
    nless_scaler = StandardScaler()
    nless_use_feature = nless_scaler.fit_transform(nless_use_feature)
    nless_reducer.fit(nless_use_feature)
    nless_embedding = nless_reducer.transform(nless_use_feature)

    nless_res_umap = pd.DataFrame({'UMAP1': nless_embedding[:,0]})
    for i in range(1, 2):
        nless_res_umap['UMAP'+str(i+1)] = nless_embedding[:,i]

    nless_res_umap.to_csv(os.path.join(res_f, save_f, "nless_all_features_projection-" +\
        datetime.now().strftime("%d%m%Y") + ".csv"), index=False)
    pickle.dump(nless_reducer, open(os.path.join(res_f, save_f, "nless_reducer-" +\
        datetime.now().strftime("%d%m%Y") + ".sav"), 'wb'))

else:
    nless_res_umap = pd.read_csv(os.path.join(res_f, save_f, "nless_all_features_projection-07072022.csv"),
        index_col=False)  

print("Projection ready! (Again).")


#%% Displays
nless_shuffle = np.random.permutation(nless_res_umap.shape[0])
nless_results = nless_res_umap.iloc[nless_shuffle]
nless_feats = nless_all_features.iloc[nless_shuffle]

for cat in ["date"]: #,"acoustic","behavior","fishing_net"]:
    plt.figure()
    sns.scatterplot(data=nless_results, x='UMAP1', y="UMAP2", hue=nless_feats[cat]).set_title(
        f'UMAP projection of {res_umap.shape[0]} clicks')
    sns.set_theme(style="ticks", font_scale=1.5)

    if cat=="date":
        handles, labels = plt.gca().get_legend_handles_labels()
        dates = [datetime.strptime(ts, "%d/%m/%Y") for ts in labels]
        order = np.argsort(dates)
        plt.legend([handles[idx] for idx in order],
                   [labels[idx] for idx in order],
                   loc='best')
    else:
        plt.legend(loc='best')
    plt.show(block=True)
print("Displays are ready.")
print("\n...Nothing left...")