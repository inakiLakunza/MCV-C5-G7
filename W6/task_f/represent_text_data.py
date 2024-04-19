
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from fasttext import load_model
from dataloader import Dataset  # Importing the Dataset class from dataloader.py

# Define FastText model parameters
fasttext_params = {
    'learning_rate': 0.1,
    'epoch': 100,
    'dim': 300,  # Dimension of word vectors
    'ws': 5,  # Size of the context window
    'min_count': 1,  # Minimum number of word occurrences
    'minn': 3,  # Min length of char ngram
    'maxn': 6,  # Max length of char ngram
    'neg': 5,  # Number of negative samples
    'loss': 'softmax'  # Loss function
}

# Define CRNN model parameters
crnn_params = {
    'input_dim': 128,
    'hidden_dim': 128,
    'output_dim': 1,
    'num_layers': 1,
    'kernel_sizes': [1, 3, 5],
    'dropout': 0.5
}

def extract_texts_and_ages(regime="train"):
    dataset = Dataset(regime=regime)
    texts_and_ages = {"texts": [], "ages": []}

    for index in range(len(dataset)):
        age_group, _, _ = dataset[index]
        # Here, you might want to extract text data from videos or other sources if available
        # For the sake of demonstration, let's assume you have some dummy text data
        text_data = f"Text data for sample {index}"
        texts_and_ages["texts"].append(text_data)
        texts_and_ages["ages"].append(age_group)

    return texts_and_ages

# Extract texts and ages for training set
train_texts_and_ages = extract_texts_and_ages(regime="train")

# Extract texts and ages for validation set
val_texts_and_ages = extract_texts_and_ages(regime="val")

# Extract texts and ages for test set
test_texts_and_ages = extract_texts_and_ages(regime="test")

# Load FastText model
fasttext_model = load_model('your_fasttext_model.bin')

# Define CRNN model
class CRNN(nn.Module):
    def __init__(self, params):
        super(CRNN, self).__init__()
        self.conv_layers = nn.ModuleList([nn.Conv1d(params['input_dim'], params['hidden_dim'], kernel_size) for kernel_size in params['kernel_sizes']])
        self.lstm = nn.LSTM(params['hidden_dim'] * len(params['kernel_sizes']), params['hidden_dim'], num_layers=params['num_layers'], batch_first=True)
        self.dropout = nn.Dropout(params['dropout'])
        self.fc = nn.Linear(params['hidden_dim'], params['output_dim'])

    def forward(self, x):
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = conv_layer(x.transpose(1, 2)).transpose(1, 2)
            conv_output = nn.functional.relu(conv_output)
            conv_output = nn.functional.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            conv_outputs.append(conv_output)
        conv_outputs = torch.cat(conv_outputs, 1)
        lstm_input = self.dropout(conv_outputs)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        output = self.fc(lstm_output)
        return output

# Initialize CRNN model
crnn_model = CRNN(crnn_params)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(crnn_model.parameters())

# Train CRNN model
for epoch in range(crnn_params['num_epochs']):
    crnn_model.train()
    for texts_batch, ages_batch in YourDataLoader(train_texts_and_ages["texts"], train_texts_and_ages["ages"]):
        # Convert texts to FastText embeddings
        embeddings_batch = torch.tensor([fasttext_model.get_sentence_vector(text) for text in texts_batch])
        embeddings_batch = embeddings_batch.unsqueeze(1)  # Add channel dimension
        ages_batch = torch.tensor(ages_batch, dtype=torch.float).unsqueeze(1)
        
        # Forward pass
        outputs = crnn_model(embeddings_batch)
        loss = criterion(outputs, ages_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluate CRNN model
crnn_model.eval()
with torch.no_grad():
    test_embeddings = torch.tensor([fasttext_model.get_sentence_vector(text) for text in test_texts_and_ages["texts"]])
    test_embeddings = test_embeddings.unsqueeze(1)
    test_outputs = crnn_model(test_embeddings)
    test_preds = torch.round(torch.sigmoid(test_outputs)).squeeze().numpy()

# Compare predicted ages with actual ages
correct_preds = np.sum(test_preds == test_texts_and_ages["ages"])
accuracy = correct_preds / len(test_texts_and_ages["ages"])
print(f"Accuracy: {accuracy}")

