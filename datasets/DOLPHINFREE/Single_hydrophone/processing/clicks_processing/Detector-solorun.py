#%% Importations
import os
import librosa
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from clicks_processing.ClickUtils import TeagerKaiser_operator, butter_pass_filter


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

    group2 = parser.add_argument_group('Detector settings')
    group2.add_argument(
        '-s', '--sound_threshold',
        type=float,
        default=0.001,
        nargs='?', 
        required=False,
        help=("Value of amplitude used after the TK filter to set a minimum "
            "value to consider a point in the detection of peaks."
            "\nDefault is 0.001.")
    )
    group2.add_argument(
        '-c', '--channel',
        type=int,
        default=0,
        nargs='?', 
        required=False,
        help=("The index of the audio channel on which "
              "the detection will be done. Should be in [0, nchannels-1]."
              "\nDefault is 0 (first channel).")
    )
    group2.add_argument(
        '-k', '--click_size_sec',
        type=float,
        default=0.001,
        nargs='?', 
        required=False,
        help=("Duration of a click in seconds. "
              "For our data it is estimated to be ~1 ms"
              "\nDefault is 0.001 sec.")
    )
    group2.add_argument(
        '-freq', '--min_freq',
        type=int,
        default=50_000,
        nargs='?', 
        required=False,
        help=("Cutoff frequency to apply with the highpass filter."
              "Default is 50 kHz (quite a strict criteria).")
    )


    # fetching arguments
    args = parser.parse_args()
    audio_file = args.audio_file
    output = args.output
    sound_threshold = args.sound_threshold
    channel = args.channel
    click_size_sec = args.click_size_sec
    min_freq = args.min_freq

    # verifying arguments
    try:
        assert (os.path.exists(audio_file)), (
            f"\nInputError [--audio_file]: Could not find file '{audio_file}'.")
        assert (os.path.exists(output)), (
            f"\nInputError [--output]: Outputs directory '{output}' does not exist. "
            "\nChange the path for the output folder or create corresponding folder.")
        assert (sound_threshold > 0), (
            f"\nInputError [--sound_threshold]: sound_threshold must be positive.")
        assert (channel >= 0), (
            f"\nInputError [--channel]: channel must be positive.")
        assert (click_size_sec > 0), (
            f"\nInputError [--click_size_sec]: click_size_sec must be positive.")
        assert (min_freq > 0), (
            f"\nInputError [--min_freq]: min_freq must be positive.")
    except Exception as e:
        print(e)
        exit()

    return (audio_file, output, sound_threshold, channel, 
            click_size_sec, min_freq, args.verbose, args.show_plot) 

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

class SimpleAutoClicksDetector:
    """
    A class to detect clicks automatically in an audio recording.

    Parameters
    ----------
    wavefile_path : string
        Path to the wavefile in which clicks will be detected.
    sound_thresh : float
        Threshold for the detection of peaks in the audio signal
        filtered (high-pass + teager-kaiser).
        Default is 0.001.
    chan : integer
        In case the audio data is not mono-channel,
        index of the channel on which the detection of clicks will be made.
        Default is 0.
    click_size_sec : float
        Duration of a click in seconds.
        Default is 0.001 sec.
    highpass_freq : int
        Cutoff frequency for the highpass filter.
        Default is 50 kHz.
    verbose : boolean
        Wether to show prints about the process or not.
        Default is False.

    Methods
    -------
    create_peaks_map():
        A function to find peaks (supposedly clicks) in an audio recording.
        Generates map_peaks, of the same shape that the audio recording
        with 0 as a base and 1's showing the positions of detections.
    """
    def __init__(
    self,
    wavefile_path=None,
    sound_thresh=0.001,
    chan=0,
    click_size_sec=0.001,
    highpass_freq=50_000,
    verbose=False):
        # fetching arguments
        if (isinstance(wavefile_path, str) and
                isinstance(sound_thresh, float) and
                isinstance(chan, int) and
                isinstance(click_size_sec, float) and
                isinstance(highpass_freq, int) and
                isinstance(verbose, bool)):
            self.wavefile_path = wavefile_path
            self.sound_thresh = sound_thresh
            self.chan = chan
            self.click_size_sec = click_size_sec
            self.highpass_freq = highpass_freq
            self.verbose = verbose
        
        else:
            print(f"{BColors.FAIL}Error in parameters.{BColors.ENDC}")
            return None

        # actual click detection
        if self.verbose:
            print("Loading audio data...")
        self.audio_data, self.sr = librosa.load(
            self.wavefile_path, sr=None, mono=False)
        if self.verbose:
            print("Finding clicks in waveform...")
        self.create_peaks_map()
        if self.verbose:
            print(f"\t{BColors.OKGREEN}Done.{BColors.FAIL}")

    def create_peaks_map(self):
        # assign parameters
        max_length = int(self.sr*self.click_size_sec)
        mini_space = int(self.sr*self.click_size_sec*2)

        # check if mono or stereo+
        if len(self.audio_data.shape) == 1:
            self.signal = np.copy(self.audio_data)
        else:
            self.signal = np.copy(self.audio_data[self.chan])
        # detect clicks
        self.signal_high = butter_pass_filter(self.signal, self.highpass_freq, self.sr, mode='high')
        self.tk_signal = TeagerKaiser_operator(self.signal_high)
        self.signal_peaks = find_peaks(self.tk_signal, 
                                distance=mini_space,
                                width=[0,max_length],
                                prominence=self.sound_thresh)[0]
        map_peaks = np.zeros(len(self.tk_signal), dtype=int)
        map_peaks[self.signal_peaks] = 1

        self.map_peaks = map_peaks


#%% Parameters
# Audio parameters
(file_path, output_path, sound_thresh, channel, 
 click_size_sec, fmin, verbose, do_plot) = fetch_inputs()


#%% Main executions
Detector = SimpleAutoClicksDetector(
    wavefile_path=file_path,
    sound_thresh=sound_thresh,
    chan=channel,
    click_size_sec=click_size_sec,
    highpass_freq=fmin,
    verbose = verbose)

# save peak locations
arg_peaks = np.nonzero(Detector.map_peaks)[0]
df = pd.DataFrame()
df["time"] = arg_peaks/Detector.sr
df["signal_amplitude"] = Detector.signal[arg_peaks]
df["TK_amplitude"] = Detector.signal[arg_peaks]
df.to_csv(
    os.path.join(output_path, os.path.basename(file_path)[:-4]+"_clicks-detection.csv"), 
    index=False)
if verbose:
    print(f"\n{BColors.OKCYAN}File saved in '{output_path}'.{BColors.ENDC}")

if do_plot:
    if verbose:
        print("Showing plot.")
    fig, ax = plt.subplots(1,1)
    ax.plot(
        np.arange(0, len(Detector.signal_high))/Detector.sr,
        Detector.signal_high,
        color='tab:blue', zorder=-1)
    ax.scatter(
        arg_peaks/Detector.sr,
        Detector.signal_high[arg_peaks],
        color='tab:orange', s=10, zorder=1)
    plt.show(block=True)


