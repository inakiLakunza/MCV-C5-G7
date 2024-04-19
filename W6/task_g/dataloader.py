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


def parse_data(LABELS_PATH):
    data = []
    for label in os.listdir(LABELS_PATH):
        if label.startswith('.'): continue
        complete_path = os.path.join(LABELS_PATH, label)
        all_paths = sorted(os.listdir(complete_path))
        for i in range(0, len(all_paths) - 1, 3):
            dict_files = {}
            for i in range(3):
                if all_paths[i].split('.')[-1] == 'jpg':
                    dict_files["image_path"] = os.path.join(LABELS_PATH, label, all_paths[i])
                elif all_paths[i].split('.')[-1] == 'wav':
                    dict_files["audio_path"] = os.path.join(LABELS_PATH, label, all_paths[i])
                elif all_paths[i].split('.')[-1] == 'pkl':
                    dict_files["text_path"] = os.path.join(LABELS_PATH, label, all_paths[i])
            dict_files["label"] = label
            data.append(dict_files)
    return data


    

class Dataset():
    def __init__(self, regime="train"):

        assert regime.lower() in ["train", "val", "test"], "Chosen regime is not correct"

        if regime == 'train':
            LABELS_PATH = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/train/'
        elif regime == 'val':
            LABELS_PATH = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/valid'
        else:
            LABELS_PATH = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/test'

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.data = parse_data(LABELS_PATH)

        self.data_train_transform = transforms.Compose([
            v2.Resize(size=224),
            v2.TrivialAugmentWide(),
            # Turn the image into a torch.Tensor
            v2.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
            # resnet50 normalization
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def read_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.data_train_transform(img)
        return img_tensor
    
    def read_text(self, text_path):
        with open(text_path, 'rb') as f:
            txt = pickle.load(f)
        return txt
    
    def read_audio(self, wav_path):
        data, sr = librosa.load(wav_path, sr=8000)
        data_noise_reduced = nr.reduce_noise(y=data, sr=8000, stationary=False)
        onsets = librosa.onset.onset_detect(
            y=data_noise_reduced, sr=8000, 
            units="time", hop_length=128, backtrack=False
        )
        duration = 15.302
        number_of_words = len(onsets)
        words_per_second = number_of_words / duration
        tempo = librosa.beat.tempo(y=data_noise_reduced, sr=8000, start_bpm=10)[0]
        f0, _, _ = librosa.pyin(y=data_noise_reduced, sr=8000,
                                fmin=10, fmax=4000, frame_length=1024)
        timepoints = np.linspace(0, duration, num=len(f0), endpoint=False)
        f0_values = [
            np.nanmean(f0),
            np.nanmedian(f0),
            np.nanstd(f0),
            np.nanpercentile(f0, 5),
            np.nanpercentile(f0, 95),
        ]
        return [words_per_second, tempo, f0_values]

    def __getitem__(self, index):
        record = self.data[index]
        return (self.read_audio(record["audio_path"]), 
                self.read_image(record["image_path"]), 
                self.read_text(record["text_path"]), 
                record["label"])
    