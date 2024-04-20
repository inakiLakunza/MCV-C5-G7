
import sys
import os

sys.path.append('./..') 

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from audio_dataloader import Dataset




class AudioModel(nn.Module):
    def __init__(self):

        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        
        # ARCHITECTURE
        #======================================
        self.input_shape = 7  # Words per second, beats per minute, 
                              # and 5 fundamental frequency attributes
        self.output_shape = 7 # 7 different classes

        self.dropout_prob = 0.4

        # 7->16->32->16->7

        # 7->16
        self.block1 = nn.Sequential(
            nn.Linear(self.input_shape, 16, 
                      bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
        ).to(self.device)

        # 16->32
        self.block2 = nn.Sequential(
            nn.Linear(16, 32, 
                      bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
        ).to(self.device)

        # 32->16
        self.block3 = nn.Sequential(
            nn.Linear(32, 16, 
                      bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
        ).to(self.device)

        # 16->7
        self.block4 = nn.Sequential(
            nn.Linear(16, self.output_shape, 
                      bias=True),
            nn.BatchNorm1d(7),
        ).to(self.device)
        #======================================


        # LOSS AND OPTIMIZER
        #======================================
        TRAIN_DATA_PER_CATEGORY: list[int] = [10, 164, 1264, 2932, 1353, 232, 51]
        n_elements_augmented: int = sum(TRAIN_DATA_PER_CATEGORY)

        class_weigths = [num/n_elements_augmented for num in TRAIN_DATA_PER_CATEGORY]
        class_weights = torch.tensor(class_weigths).to(self.device)
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(self.device))

        params = self.parameters()
        self.optimizer = torch.optim.AdamW(params, lr=1e-6, weight_decay=0.)
        #======================================



    def forward(self, x):
        # shape x: 7

        # shape out_block1: 16
        out_block1 = self.block1(x)

        # shape out_block2: 32
        out_block2 = self.block2(out_block1)

        # shape out_block3: 16
        out_block3 = self.block3(out_block2)

        # shape residual_1_3: 16
        #residual_out1_out3 = out_block1 + out_block3 

        # shape out_block4: 7
        out_block4 = self.block4(out_block3)

        # output logits, connection between input and 
        # ESTO NO SE SI ESTA BIEN, PORQUE DIRECTAMENTE VAMOS A USAR ESTE OUTPUT
        out_residual = out_block4 + x

        return out_block4
    

    def evaluate(self, dataloader: torch.utils.data.DataLoader, regime="val", stop_batch=None):
        self.eval()

        loss = 0
        accuracy = 0
        batches = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                audio_info, labels = data # (batch_size, 7) , (batch_size, 1)
                
                audio_info = audio_info.to(self.device)
                labels = labels.to(self.device)
                # We have labels from 1 to 7, but the cross
                # entropy needs them to start from 0
                labels = torch.add(labels, -1)
                
                assert labels.min() >= 0 and labels.max() <= 6, "Label indices out of range"

                # Predict 
                outputs = self(audio_info)
                pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

                # Loss
                loss += self.loss(outputs, labels)

                # Acc
                correct_predictions = (pred_class == labels).float()
                accuracy += correct_predictions.sum() / len(labels)
                batches += 1

                # WE WANT EACH EPOCH TO BE FASTER, AND WE WILL USE MORE EPOCHS
                if stop_batch:
                    if batch_idx == stop_batch: break

            average_loss = loss / batches
            average_accuracy = accuracy / batches

            #print(f'{"Validation" if regime.lower()=="val" else "Test"} Loss: {average_loss:.4f}')
            #print(f'{"Validation" if regime.lower()=="val" else "Test"} Accuracy: {average_accuracy:.4f}')

        return average_loss, average_accuracy


    def train_loop(self, train_dataloader: torch.utils.data.DataLoader, 
                    val_dataloader: torch.utils.data.DataLoader,
                    n_epochs=1, 
                    stop_batch=None) -> None:
        
        best_acc = 0.0
        best_loss = 1000000

        save_path_root = './weights/audio_model'
        os.makedirs("./weights", exist_ok=True)

        for epoch in range(n_epochs):
            self.train()
            running_loss = []
            running_acc = []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for batch_idx, data in pbar:
                audio_info, labels = data # (batch_size, 7) , (batch_size, 1)
                
                audio_info = audio_info.to(self.device)
                labels = labels.to(self.device)
                # We have labels from 1 to 7, but the cross
                # entropy needs them to start from 0
                labels = torch.add(labels, -1)
                
                assert labels.min() >= 0 and labels.max() <= 6, "Label indices out of range"

                self.optimizer.zero_grad()

                # Predict 
                outputs = self(audio_info)
                pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

                # Compute loss
                batch_loss = self.loss(outputs, labels)
                batch_loss.backward()
                self.optimizer.step()

                # Compute accuracy
                correct_predictions = (pred_class == labels).float()
                accuracy = correct_predictions.sum() / len(labels)

                running_loss.append(batch_loss.item())
                running_acc.append(accuracy)

                if stop_batch:
                    if batch_idx == stop_batch: break

            print(f'EPOCH {epoch} |  Avg Loss: {torch.Tensor(running_loss).mean()}  Avg Acc: {torch.Tensor(running_acc).mean()}')
            
            val_loss, val_acc = self.evaluate(val_dataloader, stop_batch=120)
            print(f'EPOCH {epoch} |  Val loss: {val_loss}  Avg Acc: {val_acc}\n\n')
            

            # if best val_loss until now, save it
            if val_loss < best_loss:
                # Save info of best one
                model_info_best_loss = {
                    "Epoch": epoch,
                    "Val_loss": val_loss,
                    "Val_acc": val_acc,
                    "Optimizer": self.optimizer.state_dict(),
                    "Weights": self.state_dict()
                }

                # Update
                best_loss = val_loss

                # Save
                torch.save(model_info_best_loss, save_path_root+"_best_loss_torch_load_GOOD.pth")

            
            # if best val_acc until now, save it
            if val_acc > best_acc:
                # Save info of best one
                model_info_best_acc = {
                    "Epoch": epoch,
                    "Val_loss": val_loss,
                    "Val_acc": val_acc,
                    "Optimizer": self.optimizer.state_dict(),
                    "Weights": self.state_dict()
                }

                # Update
                best_acc = val_acc

                # Save
                torch.save(model_info_best_acc, save_path_root+"_best_acc_torch_load_try_GOOD.pth")
    
    
        
        


            

        

if __name__ == '__main__':

    BATCH_SIZE = 16

    audio_model = AudioModel()
    print("Loading data...")
    # DATALOADER =========================================================
    train_dataset = Dataset(regime='train')
    val_dataset = Dataset(regime='val')
    test_dataset = Dataset(regime='test')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    #==============================================

    # TRAIN
    audio_model.train_loop(train_dataloader, val_dataloader, n_epochs=250, stop_batch=120)



    # TEST ON TEST SET
    print(f'\n\n\nFINAL TEST ==============================')
    loss, acc = audio_model.evaluate(test_dataloader)
    print(f'Test Loss: {loss} \nTest Acc: {acc}')
