import librosa
import numpy as np
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import librosa
import os

from path_constants import *

class SpeechDatasetSpectrogram(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, audio_path: str, target: str,
                 classes: dict = None, transform = None):
        self.df = df
        self.audio_path = audio_path
        self.target = target
        self.classes = classes
        self.transform = transform
        

    def __len__(self):
        return len(self.df)

    def extrair_log_mel(self, audio_path):
        if not os.path.exists(audio_path):
            return None  # Se o arquivo não existe, retorna None

        y, sr = librosa.load(audio_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec


    def padronizar_log_mel(self, log_mel, max_len=300):
        if log_mel.shape[1] > max_len:
            return log_mel[:, :max_len]
        else:
            pad_width = max_len - log_mel.shape[1]
            return np.pad(log_mel, ((0,0), (0,pad_width)), mode='constant')
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if classes:
            num_classes = len(self.classes)
            label_idx = self.classes[row[self.target]]
            label = torch.nn.functional.one_hot(torch.tensor(label_idx), num_classes).float()
        else:
            label = torch.tensor(row[target], dtype=torch.float32).unsqueeze(0)  # [1] -> [1, 1]

        
        audio_path = os.path.join(self.audio_path, row['filename'] + '.mp3')
        log_mel = self.extrair_log_mel(audio_path)
        log_mel_pad = self.padronizar_log_mel(log_mel)
        log_mel_tensor = torch.tensor(log_mel_pad.T, dtype=torch.float32)# shape: (1, 128, max_len)
    
        return log_mel_tensor, label
    
    def dataset_clean(self):
        print(f'Size before clean {len(self.df)}')
        # Remove columns that have 80% of missing (NaN) values
        self.df = self.df.dropna(axis=1, thresh=0.8*len(self.df))

        # Remove data entries that have no associated recording
        self.df = self.df[self.df['file_missing?'] != True]

        # Fix typo in 'sex' column
        self.df['sex'] = self.df['sex'].replace('famale', 'female')

        # Dataset has 2140 entries but we have only 2128 mp3 files
        existing_files = [file.split('.')[0] for file in os.listdir(RECORDINGS_PATH)]  # lista de arquivos existentes na pasta
        existing_files
        # Remove rows that have missing recordings files
        self.df = self.df[self.df['filename'].isin(existing_files)]

        print(f'Size after clean {len(self.df)}')

class AudioLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, num_layers=2, num_classes=2):
        super(AudioLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # Camadas densas com dropout e ReLU
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)     # out: (batch, time, hidden)
        out = hn[-1]                     # pega a última saída da última camada: (batch, hidden_size)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        out = self.out(out)
        out = torch.sigmoid(out)
        return out


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        print(f'train lab {labels}')
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        print(f'train out {outputs}')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #preds = outputs.argmax(dim=1)
        preds = (outputs > 0.5).int()

        print(f'preds {preds.item()}')
        print(f'labs {labels.item()}')
        correct += (preds.item() == labels.item())
        print(f'correct {correct}')
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            print(f'val lab {labels}')
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            print(f'val out {outputs}')
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # preds = outputs.argmax(dim=1)
            preds = (outputs > 0.5).int()


            print(f'preds {preds.item()}')
            print(f'labs {labels.item()}')
            correct += (preds.item() == labels.item())

            print(f'correct {correct}')
            total += labels.size(0)
            print(f'total {total}')
            print("Labels after .cpu().numpy():", labels)



            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    print(all_labels)
    print(all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return epoch_loss, epoch_acc, conf_matrix

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("./conf.png")

# Dividir treino e validação (80% treino, 20% validação)
df = pd.read_csv(DATASET_PATH)
# allowed_countries = {'usa', 'canada', 'uk', 'australia'}
# df = df[
#     (df['country'].str.lower().isin(allowed_countries)) &
#     (df['native_language'].str.lower() == 'english')
# ]
usa_df = df[df['country'] == 'usa']
non_usa_df = df[df['country'] != 'usa']
non_usa_df = non_usa_df.sample(len(usa_df), random_state=42)
df = pd.concat([non_usa_df, usa_df], ignore_index=True)

df['isUsa'] = (df['country'] == 'usa').astype(int)

target = 'isUsa'
# classes = {"usa": 0, "canada": 1, "uk": 2, "australia": 3}
classes = None
speech = SpeechDatasetSpectrogram(df,
                        RECORDINGS_PATH,
                        None)
speech.dataset_clean()



# Perform train-test split
train_df, test_df = train_test_split(speech.df, 
                                     test_size=0.2, 
                                     stratify=speech.df[target], 
                                     random_state=42)




# Train and test datasets
train_df = SpeechDatasetSpectrogram(train_df, audio_path=RECORDINGS_PATH,
                          target=target, classes=classes)
test_df = SpeechDatasetSpectrogram(test_df, audio_path=RECORDINGS_PATH, 
                        target=target, classes=classes)

print(f'train{train_df.df['isUsa'].value_counts()} teste{test_df.df['isUsa'].value_counts()}')
# DataLoaders
train_loader = DataLoader(train_df, batch_size=1, shuffle=True)
test_loader = DataLoader(test_df, batch_size=1, shuffle=False)

# Configurar modelo
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model = AudioLSTM(input_size=128,
                  hidden_size=64,
                  num_layers=2,
                  num_classes=1).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

# Loop de treinamento
epochs = 10
print("Starting train-validation loop...")
result_csv_path = os.path.join(RESULTS_PATH, 'is_usa.csv')
columns = [
    'epoch', 'time', 'train_loss', 'train_acc', 'val_loss', 'val_acc'
]
pd.DataFrame(columns=columns).to_csv(result_csv_path, index=False)
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    start = time.time()
    train_loss, train_acc = train(model, optimizer, criterion, train_loader, device)
    val_loss, val_acc, conf_matrix = evaluate(model, criterion, test_loader, device)
    end = time.time()
    elapsed_time = end- start

    # Store metrics for plotting or logging
    result = [epoch + 1, 
                elapsed_time, 
                train_loss, 
                train_acc, 
                val_loss, 
                val_acc]
    pd.DataFrame([result], columns=columns).to_csv(
        result_csv_path, mode='a', index=False, header=False
    )

    #plot_confusion_matrix(conf_matrix, list(allowed_countries))