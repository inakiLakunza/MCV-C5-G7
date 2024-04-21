
import os
import sys
sys.path.append('./..') 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import pickle
from torchvision.transforms import v2
from full_dataloader import Dataset
import tqdm
import torch
import torch.optim as optim
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import ConcatDataset, DataLoader
import librosa
import librosa.display
import noisereduce as nr
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
import sys
from model_text import TextEmbeddingLayer
from filter_datasets import FilteredImageFolder
from audio_model import AudioModel
import csv

class Model():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # MODEL FOR IMAGES
        #==========================================================
        print("Loading Inception...")
        image_model_weights = './weights/image_model.pth'
        self.image_model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=7).to(self.device)
        self.image_model.load_state_dict(torch.load(image_model_weights, map_location=self.device))
        #==========================================================

        # MODEL FOR TEXT
        #==========================================================
        print("Loading BERT...")
        text_model_weights = './weights/text_model.pth'
        self.text_model = TextEmbeddingLayer().to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.text_model.load_state_dict(torch.load(text_model_weights, map_location=self.device))
        #==========================================================

        # MODEL FOR AUDIO
        #==========================================================
        print("Loading Audio model...")
        audio_model_weights = './weights/audio_model_best_loss_torch_load_GOOD_2.pth'
        self.audio_model = AudioModel().to(self.device)
        self.audio_model.load_state_dict(torch.load(audio_model_weights, map_location=self.device)["Weights"])
        #==========================================================


    def test(self, test_dataloader):
        self.text_model.eval()
        self.image_model.eval()
        self.audio_model.eval()
        accuracy = 0
        accuracy_image = 0
        accuracy_text = 0
        accuracy_audio = 0
        batches = 0
        cont_pred=1
        predictions = [['VideoName','ground_truth','prediction']]
        csv_file='predictions_test_set.csv'
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Saltar la primera línea (encabezado)
            cont=1
            for row in csv_reader:
                predictions.append([row[0],row[1],'-1'])
        with torch.no_grad():
            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            for batch_idx, data in pbar:
                images, texts, audios, labels = data

                images = images.to(self.device)
                audios = audios.to(self.device)
                labels = labels.to(self.device)
                labels = torch.add(labels, -1)
                assert labels.min() >= 0 and labels.max() <= 6, "Label indices out of range"
       
            
         


                # PRED IMAGES ==========================================================================
                outputs_images = self.image_model(images)
                pred_images_class = torch.argmax(torch.softmax(outputs_images, dim=1), dim=1)
                accuracy_image += ((pred_images_class == labels).float().sum()) / len(labels)
                print(f'Images predicted: {pred_images_class}')

                # PRED TEXT ============================================================================
                encoding = self.tokenizer.batch_encode_plus(
                    texts,                    # List of input texts
                    padding=True,              # Pad to the maximum sequence length
                    truncation=True,           # Truncate to the maximum sequence length if necessary
                    return_tensors='pt',      # Return PyTorch tensors
                    add_special_tokens=True    # Add special tokens CLS and SEP
                )
                input_ids = encoding['input_ids'].to(self.device)  # Token IDs
                attention_mask = encoding['attention_mask'].to(self.device)  # Attention mask
                outputs = self.bert(input_ids, attention_mask=attention_mask)
                word_embeddings = outputs.last_hidden_state  # This contains the embeddings
                sentence_embedding = word_embeddings.mean(dim=1).to(self.device)  # Average pooling along the sequence length dimension
                outputs_text = self.text_model.forward(sentence_embedding)
                pred_text_class = torch.argmax(torch.softmax(outputs_text, dim=1), dim=1)
                accuracy_text += ((pred_text_class == labels).float().sum()) / len(labels)
                print(f'Texts predicted: {pred_text_class}')

                # PRED AUDIO ==========================================================================
                outputs_audio = self.audio_model(audios)
                pred_audio_class = torch.argmax(torch.softmax(outputs_audio, dim=1), dim=1)
                accuracy_audio += ((pred_audio_class == labels).float().sum()) / len(labels)
                print(f'Audios predicted: {pred_audio_class}')
                
                # PRED FINAL CLASS ====================================================================
                weight_vector=[1,1,1]
                outputs_images = weight_vector[0] * outputs_images
                outputs_text   = weight_vector[1] * outputs_text
                outputs_audio  = weight_vector[2] * outputs_audio
                
                all_outputs = (outputs_images + outputs_text + outputs_audio)
                pred_final_class = torch.argmax(torch.softmax(all_outputs, dim=1), dim=1)
                predictions[cont_pred][2] = pred_final_class.tolist()[0]+1 # y_pred_class.tolist()[0] from 0 to 6
                cont_pred +=1
                print(f'Accuracy Image: {accuracy_image}')
                print(f'Accuracy Text: {accuracy_text}')
                print(f'Accuracy Audio: {accuracy_audio}')
                print(f'Final Prediction: {pred_final_class} \n GT: {labels} \n\n')
                
                # ACCURACY ====================================================================
                correct_predictions = (pred_final_class == labels).float()
                accuracy += correct_predictions.sum() / len(labels)

                batches += 1

            average_accuracy = accuracy / batches
            average_accuracy_image = accuracy_image / batches
            average_accuracy_text = accuracy_text / batches
            average_accuracy_audio = accuracy_audio / batches

        print(f"IMAGE TEST ACCURACY = {average_accuracy_image}")
        print(f"TEXT TEST ACCURACY = {average_accuracy_text}")
        print(f"AUDIO TEST ACCURACY = {average_accuracy_audio}")
        print(f"FINAL TEST ACCURACY = {average_accuracy}")

        csv_file = 'predictions_multimodal_test_set.csv'

        np.savetxt(csv_file,
        #predictions,
        predictions,
        delimiter =",",
        fmt ='% s')


        return average_accuracy


if __name__ == '__main__':
    batch_size = 1
    model = Model()
    print("Loading data...")
    test_dataset = Dataset(regime='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    model.test(test_dataloader)


