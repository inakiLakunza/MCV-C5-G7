
import numpy as np

import matplotlib.pyplot as plt

import librosa

def plot_audio_simple(data, length=15.302, 
                      save_path = 'audio_form.png',
                      title = None):
    
    # Clear previous plot, just in case
    plt.clf()

    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title if title else "Audio signal as waverform")
    plt.savefig(save_path)


def plot_librosa_audio_simple(data, samplerate=8000,
                              save_path = 'librosa_audio_form.png',
                              title = None):

    # Clear previous plot, just in case
    plt.clf()

    plt.figure(figsize=(12, 3))

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title if title else "Audio signal as waverform")
    ax = plt.gca()
    librosa.display.waveshow(data, sr=samplerate, ax=ax, color="blue")
    plt.savefig(save_path)


def plot_librosa_audio_comparison(original_data, denoised_data, samplerate=8000, save_path='librosa_audio_comparison.png', title=None):
    plt.clf()
    plt.figure(figsize=(12, 3))
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title if title else "Original vs Denoised waveform")
    ax = plt.gca()
    librosa.display.waveshow(original_data, sr=samplerate, ax=ax, color="blue", alpha=0.7, label="Original waveform")
    librosa.display.waveshow(denoised_data, sr=samplerate, ax=ax, color="red", alpha=0.7,  label="Denoised waveform")
    plt.legend()
    plt.savefig(save_path)

def plot_librosa_audio_dual(original_data, denoised_data, samplerate=8000, save_path='librosa_audio_dual.png', title=None, title1=None, title2=None):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plt.suptitle(title if title else "Original vs Denoised waveform")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(title1 if title1 else "Original waveform")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(title2 if title2 else "Denoised waveform")
    librosa.display.waveshow(original_data, sr=samplerate, ax=ax1, color="blue", label="Original waveform")
    librosa.display.waveshow(denoised_data, sr=samplerate, ax=ax2, color="red",  label="Denoised waveform")
    plt.savefig(save_path)


def plot_frequency(data_freq, possible_freqs,
                   save_path = 'frequency_plot.png',
                   title = None):
    """
    Plot frequencies, use 'get_frequency' function
    to get the needed variables for plotting
    """

    # Clear previous plot, just in case
    plt.clf()

    plt.figure(figsize=(12, 3))
    plt.semilogx(possible_freqs[: len(possible_freqs) // 2], data_freq[: len(possible_freqs) // 2])
    plt.xlabel("Frequency (Hz)")
    plt.title(title if title else "Frequency representation of the audio")
    plt.savefig(save_path)


def plot_wave_with_onsets(data, onsets,
                          samplerate=8000,
                          save_path = 'waveform_with_onsets.png',
                          title = None):
    plt.clf()
    plt.figure(figsize=(12, 3))
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title if title else "Audio waveform with onsets")
    ax = plt.gca()
    librosa.display.waveshow(data, sr=samplerate, ax=ax, color="blue")
    for o in onsets:
        plt.vlines(o, -0.3, 0.3, colors="r", alpha=0.5)
    plt.savefig(save_path)


def plot_fundamental_frequency(data, f0, timepoints, 
                               samplerate=8000,
                               save_path = "fundamental_frequency.png",
                               title = None):
    
    # Plot fundamental frequency in spectogram plot
    plt.figure(figsize=(12, 6))
    x_stft = np.abs(librosa.stft(data))
    x_stft = librosa.amplitude_to_db(x_stft, ref=np.max)
    librosa.display.specshow(x_stft, sr=samplerate, 
                             x_axis="time", y_axis="log")
    plt.plot(timepoints, f0, color="cyan", linewidth=5)
    plt.title(title if title else "Fundamental frequency shown in Spectogram")
    plt.savefig(save_path)