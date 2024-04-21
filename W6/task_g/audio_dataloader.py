from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2
import pickle
import librosa
import librosa.display
import noisereduce as nr
import numpy as np
import torchaudio
import sys
import time

def parse_data(LABELS_PATH):
    data = []
    for label in os.listdir(LABELS_PATH):
        if label.startswith('.'): continue
        complete_path = os.path.join(LABELS_PATH, label)
        all_paths = sorted(os.listdir(complete_path))
        for path in all_paths:
            if path.split('.')[-1] == 'wav':
                data.append({"audio_path": os.path.join(LABELS_PATH, label, path), "label": label})
    return data


    

class Dataset():
    def __init__(self, regime="train", aug=None):

        assert regime.lower() in ["train", "val", "test"], "Chosen regime is not correct"

        if regime == 'train':
            LABELS_PATH = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/train/'
        elif regime == 'val':
            LABELS_PATH = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/valid'
        else:
            LABELS_PATH = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/test'

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.data = parse_data(LABELS_PATH)

    
    def __len__(self):
        return len(self.data)
        
    def read_audio(self, wav_path):
        # waveform, sample_rate = torchaudio.load(wav_path)
        #data, sr = librosa.load(wav_path, sr=8000)
        # start = time.time()
        data, sr = torchaudio.load(wav_path)
        # end = time.time()
        # print("torchaudio load: " + str(end-start))
        data = data.squeeze()

        # start = time.time()
        data_noise_reduced = nr.reduce_noise(y=data, sr=sr, stationary=False)
        # end = time.time()
        # print("data_noise_reduced load: " + str(end-start))

        # start = time.time()
        onsets = librosa.onset.onset_detect(
            y=data_noise_reduced, sr=sr, 
            units="time", hop_length=128, backtrack=False
        )        
        # end = time.time()
        # print("onsets load: " + str(end-start))


        duration = 15.302
        number_of_words = len(onsets)
        words_per_second = number_of_words / duration

        # start = time.time()
        tempo = librosa.beat.tempo(y=data_noise_reduced, sr=sr, start_bpm=10)[0] 
        # end = time.time()
        # print("tempo: " + str(end-start))

        # start = time.time()
        f0, _, _ = librosa.pyin(y=data_noise_reduced, sr=sr,
                                fmin=10, fmax=450, frame_length=4096)
        
        # TO AVOID NAN ERRORS
        if np.isnan(f0).all(): 
            return_list = [0.0]*7
            return torch.tensor(return_list).to(torch.float)

        # end = time.time()
        # print("f0 pyin: " + str(end-start))
        #timepoints = np.linspace(0, duration, num=len(f0), endpoint=False)

        # start = time.time()
        f0_values = [
            np.nanmean(f0),
            np.nanmedian(f0),
            np.nanstd(f0),
            np.nanpercentile(f0, 5),
            np.nanpercentile(f0, 95),
        ]

        new_f0_values = []
        for value in f0_values:
            if value == np.isnan:
                #print("nan found")
                new_f0_values.append(0.0)
            else: 
                new_f0_values.append(value)

        # end = time.time()
        # print("f0 values: " + str(end-start))

        #print(new_f0_values)

        audio_info = [words_per_second, tempo]
        audio_info.extend(new_f0_values)
        audio_info = torch.tensor(audio_info).to(torch.float)
        return audio_info

    def __getitem__(self, index):
        record = self.data[index]
        audio = self.read_audio(record["audio_path"])
        return (audio, torch.tensor(int(record["label"])))
    