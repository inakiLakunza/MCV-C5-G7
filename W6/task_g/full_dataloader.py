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

def parse_data(LABELS_PATH):
    data = []
    for label in os.listdir(LABELS_PATH):
        if label.startswith('.'): continue
        complete_path = os.path.join(LABELS_PATH, label)
        all_paths = sorted(os.listdir(complete_path))
        for i in range(0, len(all_paths) - 1, 3):
            dict_files = {}
            for j in range(i, i+3):
                if all_paths[j].split('.')[-1] == 'jpg':
                    dict_files["image_path"] = os.path.join(LABELS_PATH, label, all_paths[j])
                elif all_paths[j].split('.')[-1] == 'wav':
                    dict_files["audio_path"] = os.path.join(LABELS_PATH, label, all_paths[j])
                elif all_paths[j].split('.')[-1] == 'pkl':
                    dict_files["text_path"] = os.path.join(LABELS_PATH, label, all_paths[j])
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

        if regime == 'train':
            self.data_train_transform = transforms.Compose([
                v2.Resize(size=224),
                v2.TrivialAugmentWide(),
                # Turn the image into a torch.Tensor
                v2.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
                # resnet50 normalization
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.data_train_transform = transforms.Compose([
                v2.Resize(size=224),
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
        data, sr = torchaudio.load(wav_path)
        data = data.squeeze()
        data_noise_reduced = nr.reduce_noise(y=data, sr=sr, stationary=False)

        onsets = librosa.onset.onset_detect(
            y=data_noise_reduced, sr=sr, 
            units="time", hop_length=128, backtrack=False
        )        
        duration = 15.302
        number_of_words = len(onsets)
        words_per_second = number_of_words / duration

        tempo = librosa.beat.tempo(y=data_noise_reduced, sr=sr, start_bpm=10)[0] 
        f0, _, _ = librosa.pyin(y=data_noise_reduced, sr=sr,
                                fmin=10, fmax=450, frame_length=4096)
        
        # TO AVOID NAN ERRORS
        if np.isnan(f0).all(): 
            return_list = [0.0]*7
            return torch.tensor(return_list).to(torch.float)
        
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
                new_f0_values.append(0.0)
            else: 
                new_f0_values.append(value)

        audio_info = [words_per_second, tempo]
        audio_info.extend(new_f0_values)
        audio_info = torch.tensor(audio_info).to(torch.float)
        return audio_info

    def __getitem__(self, index):
        record = self.data[index]
        images = self.read_image(record["image_path"])
        texts  = self.read_text(record["text_path"])
        audios = self.read_audio(record["audio_path"])
        labels = torch.tensor(int(record["label"]))
        return (images, texts, audios, labels)
    