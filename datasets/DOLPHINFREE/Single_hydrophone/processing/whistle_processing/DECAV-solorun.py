#%% Importations
import os
import json
import argparse
import numpy as np
from librosa import load, stft, pcen, amplitude_to_db

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy import interpolate
from scipy.stats.mstats import gmean
from scipy.signal import resample

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from whistle_processing.WhistleUtils import (get_local_maxima, get_trajectories, 
    select_trajectories, sparsity_ridoff, harmonize_trajectories)


def _frame_times(n_frames, hop_length, sr):
    # librosa frame index -> seconds
    # time_of_frame_start = frame * hop_length / sr
    import numpy as _np
    return _np.arange(n_frames) * (hop_length / sr)

def _freq_axis(n_rows, n_fft, sr, fmin_bins=0):
    # rows are already trimmed by f_min in your code; reflect that
    import numpy as _np
    freqs_full = _np.linspace(0, sr/2, 1 + n_fft//2)
    return freqs_full[-n_rows:]  # matches spectrum[f_min:,:]

def _safe_slice(lo, hi, max_len):
    lo = max(0, int(lo))
    hi = min(max_len, int(hi))
    if hi <= lo:
        hi = min(max_len, lo + 1)
    return lo, hi


def save_whistle_spectrograms(
    traj_pixels,               # 2D array with integer labels (0=background)
    spectrum,                  # 2D magnitude or dB array [freq_rows, time_frames]
    out_dir,                   # folder to save images/snippets
    base_name,                 # e.g., os.path.basename(file)[:-4]
    n_fft, noverlap, sr, hop_length,
    pad_time_s=0.05,           # 50 ms padding on both sides
    pad_freq_hz=1500,          # 1.5 kHz padding
    dB=True,                   # convert to dB for plotting if spectrum is magnitude
    save_audio=False,          # also export audio
    audio_signal=None,         # signal_dec (resampled)
    fig_dpi=180
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from librosa import power_to_db, amplitude_to_db

    os.makedirs(out_dir, exist_ok=True)

    # Prepare axes
    n_rows, n_cols = spectrum.shape
    times = _frame_times(n_cols, hop_length, sr)
    freqs = _freq_axis(n_rows, n_fft, sr)

    # Convert to dB for nicer visualization (if not already)
    spec_to_plot = spectrum
    if dB:
        # your spectrum is |STFT| (amplitude). amplitude_to_db is appropriate.
        spec_to_plot = amplitude_to_db(np.maximum(spectrum, 1e-12), ref=np.max)

    # Iterate whistles
    labels = np.unique(traj_pixels)
    labels = labels[labels > 0]  # skip background

    for lbl in labels:
        fy, tx = np.where(traj_pixels == lbl)  # fy=freq-row indices, tx=time-frame indices
        if len(fy) == 0:
            continue

        # bounds in frames/freq-rows
        t_min, t_max = tx.min(), tx.max()
        f_min, f_max = fy.min(), fy.max()

        # convert padding to bins
        pad_frames = int(np.round(pad_time_s * sr / hop_length))
        # freq bin spacing (approximately)
        if len(freqs) > 1:
            df = np.mean(np.diff(freqs))
        else:
            df = (sr/2) / max(1, (n_fft//2))
        pad_bins = int(np.round(pad_freq_hz / max(df, 1e-9)))

        # apply padding in index-space
        t0, t1 = _safe_slice(t_min - pad_frames, t_max + pad_frames + 1, n_cols)
        f0, f1 = _safe_slice(f_min - pad_bins, f_max + pad_bins + 1, n_rows)

        # crop
        crop = spec_to_plot[f0:f1, t0:t1]
        if crop.size == 0:
            continue

        # Figure
        fig, ax = plt.subplots(figsize=(6, 3), dpi=fig_dpi)
        im = ax.imshow(
            crop[::-1, :],  # flip freq axis to show low->high bottom->top
            aspect='auto',
            interpolation='nearest',
            extent=[
                times[t0], times[t1-1],            # x from sec
                freqs[f0]/1000.0, freqs[f1-1]/1000.0  # y in kHz
            ]
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_title(f"Whistle {lbl}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("dB" if dB else "Amplitude")
        fig.tight_layout()

        # save PNG
        png_path = os.path.join(out_dir, f"{base_name}_whistle_{int(lbl)}.png")
        fig.savefig(png_path)
        plt.close(fig)

        # optional audio snippet spanning [t0, t1] (in the resampled signal)
        if save_audio and (audio_signal is not None):
            # frame -> sample: start_sample = t * hop_length
            start_samp = int(t0 * hop_length)
            end_samp = int(min(len(audio_signal), t1 * hop_length + n_fft))
            if end_samp > start_samp:
                import soundfile as sf
                wav_path = os.path.join(out_dir, f"{base_name}_whistle_{int(lbl)}.wav")
                sf.write(wav_path, audio_signal[start_samp:end_samp], sr)


#%% Functions
# Argparser
def fetch_inputs():
    """
    A function to fetch inputs from cmd terminal. 
    Run `$python ARGS.py -h` for more information.

    ...

    Parameters
    ----------
    None : Inputs are fetched from cmd terminal as arguments.

    Return
    ------
    See arparser's help
    """

    # setting up arg parser
    parser = argparse.ArgumentParser(
        description=
        ("This script requires LIBRARIES."
        "\nAnd it does things, TO BE DESCRIBED.")
    )

    group1 = parser.add_argument_group('File informations')
    group1.add_argument(
        '-f', '--audio_file',
        type=str,
        nargs='?', 
        required=True,
        help=("Path to the audio file that the script will use. "
            "Supports all audio formats. Determines sampling_rate automatically.")
    )
    group1.add_argument(
        '-out', '--output', 
        type=str,
        nargs='?', 
        default=os.path.join(".","outputs"),
        required=False,
        help=("Path where the contours will be saved. "
        "Contours will be saved in a .json file, different for each wavefile. "
        "Default path is './outputs'.")
    )
    group1.add_argument(
        '-new', '--resampling_rate',
        type=int,
        default=48000,
        nargs='?', 
        required=False,
        help=("Resampling rate for the Wavefile. "
        "Can be used to speed up spectrogram visualisation. "
        "Default value is '48,000' Hz.")
    )
    group1.add_argument(
        '-v', '--verbose',
        action='store_true',
        required=False,
        help=("If given, show messages about the progress of the script.")
    )
    group1.add_argument(
        '-show', '--show_plot',
        action='store_true',
        required=False,
        help=("If given, plots the results of the selection process.")
    )

    group2 = parser.add_argument_group('Spectrogram settings')
    group2.add_argument(
        '-fft', '--n_fft',
        type=int,
        default=256,
        nargs='?', 
        required=False,
        help=("Size of the fft window in samples. "
              "Default value is '256' samples.")
    )
    group2.add_argument(
        '-over', '--n_overlap',
        type=int,
        default=0,
        nargs='?', 
        required=False,
        help=("Number of fft windows overlapping [hop_length = nfft//(noverlap+1)]. "
              "Default value is '0' overlap.")
    )
    group2.add_argument(
        '-high', '--high_pass',
        type=int,
        default=4675,
        nargs='?', 
        required=False,
        help=("Frequency for the hig-pass filter. "
              "Default value is '4,675' Hz.")
    )
    group2.add_argument(
        '-p', '--pcen',
        type=bool,
        default=False,
        nargs='?',
        required=False,
        help=("If True, uses a PCEN instead of a classical spetrogram. "
              "Default is False.")
    )

    group3 = parser.add_argument_group('DECAV parameters')
    group3.add_argument(
        '-dist_f', '--distance_frequency',
        type=int,
        default=2,
        nargs='?', 
        required=False,
        help=("Maximum distance between two pixels on the frequency-axis "
              "to consider that they belong to the same whistle. "
              "Default value is '2' pixels.")
    )
    group3.add_argument(
        '-dist_t', '--distance_time',
        type=int,
        default=5,
        nargs='?', 
        required=False,
        help=("Maximum distance between two pixels on the time-axis"
              "to consider that they belong to the same whistle."
              "Default value is '5' pixels.")
    )
    group3.add_argument(
        '-nrg_r', '--energy_ratio',
        type=int,
        default=6,
        nargs='?', 
        required=False,
        help=("SNR parameter to consider that a pixel has a higher value than its neighbours."
              "Default value is '6'.")
    )
    group3.add_argument(
        '-min_s', '--min_size',
        type=float,
        default=53.3,
        nargs='?', 
        required=False,
        help=("Minimum size (in ms) to keep a whistle."
              "Default value is '53.3' ms.")
    )
    group3.add_argument(
        '-min_a', '--min_acc',
        type=float,
        default=0.5,
        nargs='?', 
        required=False,
        help=("Minimum acceleration of whistle trajectory to be kept."
              "Default value is '0.5' Hz.s-2.")
    )
    group3.add_argument(
        '-max_a', '--max_acc',
        type=float,
        default=3,
        nargs='?', 
        required=False,
        help=("Maximum acceleration of whistle trajectory to be kept."
              "Default value is '3' Hz.s-2.")
    )
    group3.add_argument(
        '-cc', '--correlation_coef',
        type=float,
        default=0.5,
        nargs='?', 
        required=False,
        help=("Limit of correlation for two whistles overlapping in time."
              "If correlation > cc then the highest frequency whistle is"
              "considered to be a harmonic, and therefore discarded."
              "Default value is '0.5'.")
    )
    group3.add_argument(
        '-s', '--sparsity_coef',
        type=float,
        default=0.5,
        nargs='?', 
        required=False,
        help=("Cleaning value. "
              "Contours with more than 'sparsity_coef'%% missing pixels are discarded."
              "Default value is '0.5'."),
    )

    # fetching arguments
    args = parser.parse_args()
    audio_file = args.audio_file
    resampling_rate = args.resampling_rate
    n_fft = args.n_fft
    n_overlap = args.n_overlap
    high_pass = args.high_pass
    output = args.output
    distance_frequency = args.distance_frequency
    distance_time = args.distance_time
    energy_ratio = args.energy_ratio
    min_size = args.min_size
    min_acc = args.min_acc
    max_acc = args.max_acc
    correlation_coef = args.correlation_coef
    sparsity_coef = args.sparsity_coef

    # verifying arguments
    try:
        assert (os.path.exists(audio_file)), (
            f"\nInputError: Could not find file '{audio_file}'.")
        assert (os.path.exists(output)), (
            f"\nInputError: Outputs directory '{output}' does not exist. "
            "\nChange the path for the output folder or create corresponding folder.")
        assert (n_overlap >= 0), (
            f"\nInputError: n_overlap can not be negative.")
        assert (distance_frequency > 0), (
            f"\nInputError: distance_frequency can not be negative.")
        assert (distance_time > 0), (
            f"\nInputError: distance_time can not be negative.")
        assert (energy_ratio > 0), (
            f"\nInputError: energy_ratio can not be negative.")
        assert (min_size > 0), (
            f"\nInputError: min_size can not be negative.")
        assert (min_acc >= 0), (
            f"\nInputError: min_acc can not be negative.")
        assert (max_acc > 0), (
            f"\nInputError: max_acc can not be negative.")
        assert (min_size >= max_acc), (
            f"\nInputError: min_size must be inferior to max_acc.")
        assert (correlation_coef >= 0) and (correlation_coef <= 1), (
            f"\nInputError: correlation_coef must be in [0, 1].")
        assert (sparsity_coef >= 0) and (sparsity_coef <= 1), (
            f"\nInputError: sparsity_coef must be in [0, 1].")
        assert (high_pass < (resampling_rate/2)), (
            f"\nInputError: high_pass frequency too high "
            "(max is resampling_rate/2).")
    except Exception as e:
        print(e)
        exit()

    return (audio_file, resampling_rate, n_fft, n_overlap, high_pass, output, 
            distance_frequency, distance_time, distance_frequency, min_size,
            min_acc, max_acc, correlation_coef, sparsity_coef, args.pcen, 
            args.verbose, args.show_plot)   

# Main functions
class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Plotting function
def plot_spectrums(list_of_spectrums, cmaps=[], direction=-1, title="Whistles", 
    bins=1, titles=[], ylabels=[]):
    n = len(list_of_spectrums)
    fig, axs = plt.subplots(nrows=n, sharex=True, sharey=True, figsize=(15,15))

    if len(cmaps) == 0:
        cmaps = ['gray_r']*n

    for i in range(n):
        axs[i].imshow(list_of_spectrums[i][::direction], aspect='auto', 
            interpolation='nearest', cmap=cmaps[i])
        axs[i].set_yticklabels([])
        if len(titles) > 0:
            axs[i].set_title(titles[i], fontsize=15)
        if len(ylabels) >0:
            axs[i].set_ylabel(ylabels[i], fontsize=12)
        axs[i].tick_params(axis='both', which='both', labelsize=10)
    axs[n-1].set_xlabel(f"Time in bins (1 bin = 1/{bins} sec)", fontsize=12)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axs

def array_to_dict(spectrogram_pixels, fft, overlap, sr):
    """
    Function to store pixels coordinates into a dict

    Args:
        spectrogram_pixels (numpy array)
        fft (int): 
            Size of the fft window in samples
        overlap (int):
            Number of fft window overlapping
        sr (int):
            Sampling rate of the audio file
            

    Returns:
        dict: 
            Dictionnary with a list of coordinates associeted to integers
            Each integer is a different whistle
    """
    values = np.unique(spectrogram_pixels)[1:]
    dict_traj = {}
    frequencies = np.arange(0, 1 + fft / 2) * sr / fft
    frequencies = frequencies[-spectrogram_pixels.shape[0]:]
    for key, value in enumerate(values):
        coords = [np.where(spectrogram_pixels == value)[1].tolist(),
                  np.where(spectrogram_pixels == value)[0].tolist()]
        # convert to freq
        coords[0] = [(coord*(fft/sr))/(overlap+1) for coord in coords[0]]
        # convert to time
        coords[1] = [frequencies[coord] for coord in coords[1]]

        dict_traj[key+1] = coords

    return dict_traj


#%% Parameters
# Audio parameters
(file, new_sr, nfft, noverlap, fmin, output, dist_f, dist_t, nrg_r, 
 taille_min, min_acce, max_acce, min_r_coef, sparsity, do_pcen, 
 verbose, plot) = fetch_inputs()

f_min = round(fmin/(new_sr/nfft))
hop_length = nfft//(noverlap+1) 
taille_traj_min = round(taille_min*(new_sr/hop_length)/1000)


#%% Main execution
# get audio data
if verbose:
    print("Loading audio file...")
signal, fe = load(file, sr=None)
duration = len(signal)/fe
# resample to decrease computation time
signal_dec = resample(signal, int((duration*new_sr)))
# extract spectral informations
Magnitude_audio = stft(signal_dec, n_fft=nfft, hop_length=hop_length)
if do_pcen:
    spectrum = pcen(np.abs(Magnitude_audio) * (2**31), bias=10)[f_min:,:]
else:
    spectrum = np.copy(np.abs(Magnitude_audio[f_min:,:]))
if verbose:
    print(f"{BColors.OKGREEN}\tDone.{BColors.ENDC}\n")

# Selection algorithm
if verbose:
    print("Searching for local maxima...")
max_loc_per_bin_check1 = get_local_maxima(spectrum, spectrum, nrg_r)[1]
if verbose:
    print("Finding contours...")
trajectories = get_trajectories(max_loc_per_bin_check1, dist_f=dist_f, dist_t=dist_t)
if verbose:
    print("Cleaning contours...")
final_traj = select_trajectories(trajectories, taille_traj_min, min_acce, max_acce, verbose=0)
corrected_traj = sparsity_ridoff(final_traj, error_thresh=sparsity)
if verbose:
    print("Removing harmonics...")
harmonized_traj = harmonize_trajectories(corrected_traj, min_r=min_r_coef, 
	min_common=taille_traj_min*2, delete=True)
traj_for_plots = harmonized_traj

out_images_dir = os.path.join(output, os.path.splitext(os.path.basename(file))[0] + "_whistles")
save_whistle_spectrograms(
    traj_pixels=traj_for_plots,
    spectrum=np.copy(spectrum),   # use magnitude; function converts to dB for plotting
    out_dir=out_images_dir,
    base_name=os.path.splitext(os.path.basename(file))[0],
    n_fft=nfft,
    noverlap=noverlap,
    sr=new_sr,
    hop_length=hop_length,
    pad_time_s=0.05,
    pad_freq_hz=1500,
    dB=True,
    save_audio=False,             # set True if you also want audio snippets
    audio_signal=signal_dec       # needed only if save_audio=True
)
if verbose:
    print(f"{BColors.OKGREEN}\tSaved whistle crops to: {out_images_dir}{BColors.ENDC}")
if verbose:
    print(f"{BColors.OKGREEN}\tContours ready!{BColors.ENDC}\n")


if plot:
    if verbose:
        print(f"Showing results...\n")

    # generate bright colors to differenciate trajectories
    prism = cm.get_cmap('prism', 256)
    newcolors = prism(np.linspace(0, 1, np.unique(harmonized_traj).shape[0]))
    pink = np.array([0/256, 0/256, 0/256, 1])
    newcolors[0, :] = pink
    newcmp = ListedColormap(newcolors)

    # By default shows the whole audio
    start = 0
    stop = spectrum.shape[1]

    # Create figure
    fig, axs = plot_spectrums([amplitude_to_db(spectrum), 
                        np.copy(max_loc_per_bin_check1), 
                        np.copy(final_traj), 
                        np.copy(harmonized_traj)], 
                ['gray_r', 'gray', newcmp, newcmp], 
                titles=['Spectrogram (dB scale)', 'Local maxima selection', 'Extraction of continuous trajectories',
                'Exclusion of harmonics'], 
                ylabels=["Frequency"]*4,
                bins=375, title="")
    # Update view
    axs[3].set_xlim(start,stop)
    plt.show(block=True)


# Saving results
if verbose:
    print("Saving .json files...")
dict1 = array_to_dict(final_traj, nfft, noverlap, new_sr)
dict2 = array_to_dict(corrected_traj, nfft, noverlap, new_sr)
dict3 = array_to_dict(harmonized_traj, nfft, noverlap, new_sr)
names = ["DECAV-results","DECAV-results-clean","DECAV-results-deharmonized"]

for i, dict_ in enumerate([dict1, dict2, dict3]):
    with open(os.path.join(output, os.path.basename(file)[:-4]+f"_{names[i]}.json"), "w") as f:
        json.dump(dict_, f, indent=4)
if verbose:
    print(f"{BColors.OKGREEN}It's done.{BColors.ENDC}")