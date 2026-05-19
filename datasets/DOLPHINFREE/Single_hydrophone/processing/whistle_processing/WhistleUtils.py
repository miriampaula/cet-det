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

from scipy.signal import butter
from scipy.stats.mstats import gmean
from scipy.spatial import distance
from scipy import interpolate

import matplotlib.pyplot as plt

#%% Functions process
def import_csv(csv_name, folder, useless=True, slash="/"):
    """
    Parameters
    ----------
    csv_name : STRING
        Name of the csv file that needs to be imported
    folder : STRING, optional
        Path to the folder containing csv file. 
        The default is data_folder (see Parameters section).
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
    with open(folder + slash + csv_name, newline="") as csv_file:
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
    data = import_csv(csv_names[0], csv_folder, slash=slash)
    for i in range(1,len(csv_names)):
        data = data + import_csv(csv_names[i], csv_folder, slash=slash)[1:]
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

#%% Trajectories algorithms
def get_local_maxima(spectrum, spectrum2, hardness, threshold=10e-5):
    """
    Parameters
    ----------
    spectrum : NUMPY ARRAY 
        Spectrogram (float values) of an audio signal. 
    hardness : INT or FLOAT
        Number of times a value has to be above the geometric mean in order to be kept.

    Returns
    -------
    local_max1 : NUMPY ARRAY
        Spectrogram with 1 & 0 indicating local maxima
    local_max2 : NUMPY ARRAY
        Spectrogram with 1 & 0 indicating local maxima above hardness*geom_mean
    """
    local_max1 = np.zeros(spectrum.shape, dtype=int)
    local_max2 = np.zeros(spectrum.shape, dtype=int)
    geom_mean = gmean(gmean(spectrum, axis=1))
    geom_mean0 = gmean(spectrum, axis=0)
    geom_mean1 = gmean(spectrum, axis=1)
    # geom_mean0, geom_mean1 = gmean(spectrum), gmean(spectrum)

    for each_bin in range(spectrum.shape[1]):
        for freq in range(1, spectrum.shape[0]-1):
            if (spectrum[freq, each_bin] > spectrum2[freq-1, each_bin]) and \
            (spectrum2[freq, each_bin] > spectrum2[freq+1, each_bin]):
                local_max1[freq, each_bin] = 1
                if (spectrum[freq, each_bin] > (threshold)):
                    if (spectrum[freq, each_bin] > (geom_mean0[each_bin]*hardness)):
                        if (spectrum[freq, each_bin] > (geom_mean1[freq]*hardness)):
                            local_max2[freq, each_bin] = 1

    return local_max1, local_max2

def get_trajectories(spectro_local_max, dist_f, dist_t):
    """
        Parameters
    ----------
    spectro_local_max : NUMPY ARRAY 
        Matrix of 1 & zeros indicating the presence of local maxima in a spectrum.
    dist_f : INT 
        Number of bins : Tolerance for frequency jumps in a trajectory.
    dist_t : INT
        Number of bins : Tolerance for time jumps in a trajectory.

    Returns
    -------
    traj : NUMPY ARRAY
        Array of int where each value (execpt 0) is a different trajectory. 
    """
    traj = np.zeros(spectro_local_max.shape, dtype=int)
    # initialisation boucle
    tr = np.where(spectro_local_max[:,0] == 1)[0]
    traj[tr,0] = np.arange(1, len(tr)+1)
    traj_max = len(tr)
    nb_traj = np.ones(len(tr), dtype=int)

    for j in range(1,spectro_local_max.shape[1]):
        tr = np.nonzero(spectro_local_max[:,j])[0]

        for k in range(len(tr)):
            coord1_dwn = max(0, tr[k]-dist_f)
            coord1_up = min(tr[k]+dist_f, spectro_local_max.shape[1])
            coord2_dwn = max(0, j-dist_t) 
            coord2_up = j-1
            traj_p = traj[coord1_dwn:(coord1_up+1), coord2_dwn:(coord2_up+1)]
            traj_p_list = traj_p[traj_p!=0]

            if len(traj_p_list) == 0:
                traj_max = traj_max+1
                traj[tr[k],j] = traj_max
                nb_traj = np.append(nb_traj, 1)
            else:
                # add: look for closest (and not highest valued) trajectory
                # if len(np.unique(traj_p_list)) <= 1:
                #     pos = np.argmax(nb_traj[traj_p_list-1])
                # else:
                #     coordx, coordy = np.where(traj_p!=0)
                #     coords = np.array([coordx, coordy]).T
                #     distances = distance.cdist(coords, np.array([traj_p.shape])/2)
                #     closests = np.where(distances==min(distances))[0]
                #     pos = np.argmax(nb_traj[traj_p[coordx[closests],coordy[closests]]-1])
                pos = np.argmax(nb_traj[traj_p_list-1])
                traj[tr[k],j] = traj_p_list[pos]
                nb_traj[traj_p_list[pos]-1] = nb_traj[traj_p_list[pos]-1]+1

        so = np.sort(traj[tr,j])
        doublon = so[np.where(so[1:]==so[:-1])[0]]
        if len(doublon) > 0:
            for k in range(len(doublon)):
                pos_doublon = np.where(traj[:,j]==doublon[k])[0]
                pos_ex = np.where(traj[:, max(0, j-dist_t):j]==doublon[k])[0]
                pos = np.argsort(np.abs(pos_doublon-pos_ex[-1]))
                traj[pos_doublon[pos[-1]],j] = traj_max+1
                nb_traj = np.append(nb_traj, 1)
                traj_max = traj_max+1
                nb_traj[doublon[k]] = nb_traj[doublon[k]]+1
    return traj

def select_trajectories(traj_matrix, min_len_traj, min_acceleration, max_acceleration, verbose=1):
    """
        Parameters
    ----------
    traj_matrix : NUMPY ARRAY
        Array of int where each value (except 0) is a different trajectory.
    min_len_traj : INT 
        Minimal length of a trajectory to be kept.
    min_acceleration : FLOAT
        Minimal acceleration for mean_acceleration over the trajectory.
    max_acceleration : FLOAT
        Minimal acceleration for max_acceleration over the trajectory.
    verbose : [0, 1]
        TMTC

        Returns
    -------
    sel_traj : NUMPY ARRAY
        Array of int where each value (execpt 0) is a selected trajectory. 
    """
    sel_traj = np.zeros(traj_matrix.shape, dtype=int)

    # first selection based on length (for optimization)
    values, counts = np.unique(traj_matrix, return_counts=True)
    keep_values = values[counts > int(min_len_traj*0.5)]

    for i in keep_values:
        if verbose:
            print(f"\tChecking trace traj {i+1}/{max(keep_values)}", end='\r')
        x, y = np.where(traj_matrix==i)
        # length verification
        taille_traj = len(x)

        if taille_traj > min_len_traj:
            vitesse = np.diff(np.array([x,y]))[0]/np.diff(np.array([x,y]))[1]
            acce = np.abs(np.diff(vitesse)/(y[2:]-y[:-2]))
            acce_avg = np.mean(acce)
            acce_max = np.max(acce)

            if (acce_avg < min_acceleration) and (acce_max < max_acceleration):
                sel_traj[np.where(traj_matrix==i)] = i
    return sel_traj

def sparsity_ridoff(trajectories, error_thresh=0.5):
    """
        Parameters
    ----------
    trajectories : NUMPY ARRAY
        Array of int where each value (except 0) is a different trajectory.
    error_tresh : FLOAT
        Proportion of the data that has to be continuous

        Returns
    -------
    corr_traj : NUMPY ARRAY
        Array of int where each value (execpt 0) is a trajectory with errors < error_thresh. 
    """
    corr_traj = np.zeros(trajectories.shape, dtype=int)
    for k,i in enumerate(np.unique(trajectories)[1:]):
        x, y = np.where(trajectories==i)
        x = x[np.argsort(y)]
        y = y[np.argsort(y)]
        time_errors = np.sum(y[1:]-y[:-1]-1)
        frequency_change = np.abs(x[1:]-x[:-1])
        frequency_errors = np.sum(frequency_change[frequency_change !=0]-1)
        if (time_errors/len(np.unique(y)) < error_thresh) and (frequency_errors/len(np.unique(y)) < error_thresh):
            corr_traj[np.where(trajectories==i)] = k
    return corr_traj

def harmonize_trajectories(trajectories, min_r, min_common=18, delete=False):
    """
        Parameters
    ----------
    trajectories : NUMPY ARRAY
        Array of int where each value (except 0) is a different trajectory.
    min_r : FLOAT
        Minimal R squared coef of regression to consider two trajectory as correlated
    min_common : INT, optional
        Minimal length that 2 trajectories must have in common to be correlated
    delete : BOOLEAN, optional
        If True, returns trajectory array without detected harmonics. 
        If False, return trajectory array with detected harmonics grouped. 

        Returns
    -------
    harmonized_trajectories : NUMPY ARRAY
        Array of int where each value (execpt 0) is a trajectory grouped with its harmonics/ without its harmonics 
    """
    values = np.unique(trajectories)[1:]
    starts = np.zeros(len(values), dtype=int)
    stops = np.zeros(len(values), dtype=int)
    for i,trace in enumerate(values):        
        y = np.where(trajectories == trace)[1]
        starts[i] = min(y)
        stops[i] = max(y)

    overlaps = [0]*len(starts)
    for i in range(len(starts)):
        overlaps_before = np.where(starts < stops[i])[0]
        overlaps_after = np.where(stops > starts[i])[0]
        overlaps[i] = np.intersect1d(overlaps_before, overlaps_after)

    whistles = np.arange(1,len(starts)+1, dtype=float)
    for overlap in overlaps:
        if len(overlap) > 1:
            tr1 = overlap[0]
            for tr2 in overlap[1:]:
                trace1 = np.where(trajectories == values[tr1])
                trace2 = np.where(trajectories == values[tr2])

                common_time = np.intersect1d(trace1[1], trace2[1])
                if len(common_time) > min_common:
                    common_interpolation = np.arange(min(common_time), max(common_time)+1)

                    trace1_sort = np.argsort(trace1[1])
                    trace10, trace11 = trace1[0][trace1_sort], trace1[1][trace1_sort]
                    trace2_sort = np.argsort(trace2[1])
                    trace20, trace21 = trace2[0][trace2_sort], trace2[1][trace2_sort]

                    f1 = interpolate.interp1d(trace11, trace10)
                    f2 = interpolate.interp1d(trace21, trace20)
                    new_trace1 = f1(common_interpolation)
                    new_trace2 = f2(common_interpolation)

                    r_coef = np.corrcoef(new_trace1, new_trace2)[0,1]
                
                else:
                    r_coef = 0

                if r_coef > min_r:
                    whistles[tr2] = whistles[tr1]+(np.random.random()*0.9)

    harmonized_trajectories = np.copy(trajectories).astype(float)
    for i, value in enumerate(values):
        harmonized_trajectories[np.where(trajectories == value)] = whistles[i]

    if delete: # will need optimization
        deharmonized_trajectories = np.copy(harmonized_trajectories)
        values = np.unique(harmonized_trajectories)[1:]
        repeated_values = np.unique(values.astype(int))
        for value in repeated_values:
            harmonics = np.where(values.astype(int)==value)[0]
            name_trajs = values[harmonics]
            lengths = np.zeros(name_trajs.shape)
            for n,name in enumerate(name_trajs):
                lengths[n] = len(np.where(harmonized_trajectories == name)[0])
            name_trajs = np.delete(name_trajs, np.argmax(lengths))
            for name in name_trajs:
                deharmonized_trajectories[np.where(harmonized_trajectories==name)]=0
        return deharmonized_trajectories

    return harmonized_trajectories

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
    cat_list : LIST OF STRINGS
        A list of categories corresponding to inputs.

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


def plot_spectrums(list_of_spectrums, cmaps=[], direction=-1, title="Whistles", 
    bins=1, titles=[], ylabels=[]):
    """
    Parameters
    ----------
    list_of_spectrums : LIST OF NUMPY ARRAYS
        List of images that will be plotted, should be a spectrogram
        or a binary image (with 0s and 1s). 
        All arrays should be of the same shape.
    cmaps : LIST
        List that must contain the same number of elements as list_of_spectrums.
        Each element is either the name of a cmap (e.g. "viridis") or a custom cmap.
    direction : INT, optionnal
        Must be 1 or -1. If -1 it inverses the order for frequency display.
        Default is -1.
    title : STRING, optionnal
        Title that will be displayed on top of the plot.
    bins : FLOAT, optionnal
        Number of bins (windows in spectrogram) per second
    titles : LIST OF STRINGS
        List that must contain the same number of elements as list_of_spectrums.
        Titles associated with each spectrogram in plot.
    ylabels : LIST OF STRINGS
        List that must contain the same number of elements as list_of_spectrums.
        Name of y-axis for each spectrogram.

    Returns
    -------
    fig, axs : MATPLOTLIB OBJECT
        figure and axis object that contain the plot of all given spectrogram.

    """
    n = len(list_of_spectrums)
    fig, axs = plt.subplots(nrows=n, sharex=True, sharey=True, figsize=(15,15))

    if len(cmaps) == 0:
        cmaps = ['gray_r']*n

    for i in range(n):
        axs[i].imshow(list_of_spectrums[i][::direction], aspect='auto', 
            interpolation='nearest', cmap=cmaps[i])
        axs[i].set_yticklabels([])
        if len(titles) > 0:
            axs[i].set_title(titles[i], fontsize=30)
        if len(ylabels) >0:
            axs[i].set_ylabel(ylabels[i], fontsize=25)
        axs[i].tick_params(axis='both', which='both', labelsize=20)
    if bins !=1 :
        axs[n-1].set_xlabel(f"Time in bins (1 bin = 1/{bins} sec)", fontsize=25)
    else:
        axs[n-1].set_xlabel(f"Time in bins)", fontsize=25)
    fig.suptitle(title)
    fig.tight_layout(pad=1)
    return fig, axs