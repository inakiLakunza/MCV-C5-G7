import sys
import os

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import scipy

import librosa
import librosa.display

import noisereduce as nr


from utils import *


def get_frequency(data, samplerate=8000):
    """
    Return FFT data and all possible frequencies (for plotting)
    """


    # Apply Fast Fourier Transform to the signal, and use absolute values
    data_freq = np.abs(scipy.fftpack.fft(data))

    # Establish all possible frequencies (for plotting)
    # (This depends on the sampling rate (HARDCODED TO 8000, 
    # THE WAY THE TEACHER DID), and the length of the signal
    # (ALL AUDIOS HAVE SAME LENGTH: 15.302s))
    possible_freqs = np.linspace(0, samplerate, len(data_freq))

    return data_freq, possible_freqs




# DATA SHAPE IS ALWAYS (OR IT SEEMS TO BE): (122416, )
# AND THE LENGTH OF THE AUDIOS:             15.302s
# AND THE USED SAMPLERATE:                  8000

if __name__ == '__main__':


    # WE CAN EXTRACT THE NUMBER OF WORDS IN THE AUDIO,
    # ITS DURATION (ALWAYS THE SAME), THE NUMBER OF WORDS PER SECOND,
    # THE TEMPO, AND THE DIFFERENT VALUES OF THE FUNDAMENTAL FREQUENCY:
    # THE MEAN, THE MEDIAN, THE STANDARD DEVIATION,
    # AND THE 5- AND 95-PERCENTILES


    SAMPLERATE = 8000
    AUDIO_LENGTH = 15.302 # seconds

    # TRY LOADING AN AUDIO AN PLOTTING ITS WAVEFORM:
    #========================================================
    # #filename = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/test/1/-9BZ8A9U7TE.000.wav'
    # filename = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/valid/6/_GFregyrwfo.000.wav'
    # samplerate, data = wavfile.read(filename)
    # print(f"data shape = {data.shape}") # (122416, )
    # length = data.shape[0] / samplerate
    # print(f"length = {length}s") # 15.302s
    # plot_audio_simple(data)
    #========================================================

    # PLOT WAVEFORM WITH LIBROSA
    #========================================================
    # filename = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/valid/6/_GFregyrwfo.000.wav'
    # data, sr = librosa.load(filename, sr=SAMPLERATE) # sr=8000

    # plot_librosa_audio_simple(data)
    #========================================================


    # PLOT FREQUENCIES (FFT TRANSFORM)
    #========================================================
    # filename = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/valid/6/_GFregyrwfo.000.wav'
    # data, sr = librosa.load(filename, sr=SAMPLERATE) # sr=8000

    # # Get frequency info
    # data_freq, possible_freqs = get_frequency(data)
    # plot_frequency(data_freq, possible_freqs)
    #========================================================


    # PLOT NORMAL vs DENOISED COMPARISONS
    #========================================================
    # filename = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/valid/6/_GFregyrwfo.000.wav'
    # data, sr = librosa.load(filename, sr=SAMPLERATE) # sr=8000

    # data_noise_reduced = nr.reduce_noise(y=data, sr=SAMPLERATE, stationary=False)

    # plot_librosa_audio_comparison(data, data_noise_reduced)
    # plot_librosa_audio_dual(data, data_noise_reduced)
    #========================================================


    # TRIMMING (CORTAR ESQUINAS)
    # WE HAVE DECIDED NOT TO USE IT, BECAUSE FOR INSTANCE,
    # CHILDREN AND VERY OLD PEOPLE NEED MORE TIME TO START
    # SPEAKING, OR MAYBE LESS, WE THINK THAT THIS INFORMATION
    # IS CHARACTERISTIC FOR OUR TASK, SO WE WILL NOT BE USING IT



    # FEATURE EXTRACTION
    #========================================================
    filename = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/valid/6/_GFregyrwfo.000.wav'
    #filename = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/valid/2/117jrwvl2Nc.004.wav'

    #filename = "/ghome/group07/C5-W6/First_Impressions_v3_multimodal/train/4/_0bg1TLPP-I.005.wav"
    #filename = "/ghome/group07/C5-W6/First_Impressions_v3_multimodal/train/1/2KSBoJZMcMU.003.wav"
    #filename = "/ghome/group07/C5-W6/First_Impressions_v3_multimodal/train/7/3Vr5-zedeWk.001.wav"

    data, sr = librosa.load(filename, sr=SAMPLERATE) # sr=8000

    data_noise_reduced = nr.reduce_noise(y=data, sr=SAMPLERATE, stationary=False)
    
    # ONSETS
    onsets = librosa.onset.onset_detect(
        y=data_noise_reduced, sr=SAMPLERATE, 
        units="time", hop_length=128, backtrack=False
    )

    #plot_wave_with_onsets(data_noise_reduced, onsets, save_path="waveform_with_onsets.png")
    #plot_wave_with_onsets(data_noise_reduced, onsets, save_path="waveform_with_onsets_2.png")

    # LENGTH
    # the duration is the same for all audios
    duration = AUDIO_LENGTH # 15.302s

    onsets = librosa.onset.onset_detect(
        y=data_noise_reduced, sr=SAMPLERATE, 
        units="time", hop_length=128, backtrack=False
    )

    number_of_words = len(onsets)
    words_per_second = number_of_words / duration
    print(f"""
    The audio signal is {duration:.3f} seconds long\n
    with an average of {words_per_second:.2f} words per second.
    """)


    # TEMPO
    tempo = librosa.beat.tempo(y=data_noise_reduced, sr=SAMPLERATE, start_bpm=10)[0]
    print(f"\n\nThe audio signal has a speed of {tempo:.2f} Beats per Minute (bpm)")



    # FUNDAMENTAL FREQUENCY
    # Lowest frequency at which a periodic sound appears (pitch in music).
    # In the spectogram plots, the fundamental frequency is the lowest 
    # horizontal strip. And the repetition of the strip pattern 
    # above this fundamental are called harmonics

    # Extract fundamental frequency using a probabilistic approach
    f0, _, _ = librosa.pyin(y=data_noise_reduced, sr=SAMPLERATE,
                            fmin=10, fmax=4000, frame_length=1024)
    
    # Establish timepoint of f0 signal
    timepoints = np.linspace(0, duration, num=len(f0), endpoint=False)

    

    f0_values = [
        np.nanmean(f0),
        np.nanmedian(f0),
        np.nanstd(f0),
        np.nanpercentile(f0, 5),
        np.nanpercentile(f0, 95),
    ]

    print("""\n
    This audio signal has a mean of {:.2f},\n
    a median of {:.2f},\n
    a Standard Deviation of {:.2f},\n
    a 5-percentile at {:.2f}\n
    and a 95-percentile at {:.2f}
    """.format(*f0_values))

    plot_fundamental_frequency(data_noise_reduced, f0, timepoints)
    #plot_fundamental_frequency(data_noise_reduced, f0, timepoints, save_path="f0_3.png")
    #========================================================

    

    

    


    

    