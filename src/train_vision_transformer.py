import os
from torch.utils.data import Dataset
import torch
import librosa
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import random
import glob
import torch.optim.lr_scheduler as lr_scheduler
from scipy.signal import butter, lfilter
from models.vit_model import VisionTransformer
from multiprocessing import cpu_count


class AudioDatasetForViT(Dataset):
    def __init__(self, ai_directory, human_directory, sr=16000, duration=3, augment=True):
        self.ai_files = glob.glob(os.path.join(ai_directory, '*.mp3'))
        self.human_files = glob.glob(os.path.join(human_directory, '*.mp3'))

        self.all_files = self.ai_files + self.human_files
        self.labels = [0] * len(self.ai_files) + [1] * len(self.human_files)
        self.sr = sr
        self.duration = duration
        self.augment = augment

        # Hardcoded global mean and std values
        self.global_mean = -58.18715250929163
        self.global_std = 15.877255962380845 

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        while True:
            audio_path = self.all_files[idx]
            label = self.labels[idx]
            try:
                y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
                y = librosa.util.fix_length(y, size=self.sr * self.duration)

                if self.augment:
                    if random.random() < 0.05:
                        y = self.apply_augmentation(y)
                    #y = self.apply_augmentation(y)

                # Clamp the audio signal to avoid large values
                y = np.clip(y, -1.0, 1.0)

                # Compute the STFT
                S = np.abs(librosa.stft(y))**2

                # Add a small constant to avoid log of zero
                S_db = librosa.power_to_db(S + 1e-10, ref=np.max)

                # Normalize using hardcoded global mean and std
                S_db = (S_db - self.global_mean) / self.global_std

                # Ensure consistent dimensions (e.g., 1025 x 94)
                target_shape = (1025, 94)
                if S_db.shape != target_shape:
                    S_db = np.pad(S_db, (
                        (0, max(0, target_shape[0] - S_db.shape[0])), 
                        (0, max(0, target_shape[1] - S_db.shape[1]))
                    ), mode='constant', constant_values=-self.global_mean)
                    S_db = S_db[:target_shape[0], :target_shape[1]]

                spectrogram_tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)
                
                return spectrogram_tensor, label, audio_path
            except Exception as e:
                print(f"Skipping file {audio_path} due to error: {e}")
                idx = (idx + 1) % len(self.all_files)


    def apply_augmentation(self, y):
        if random.random() < 0.5:
            rate = np.random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y=y, rate=rate)

        if random.random() < 0.5:
            steps = np.random.randint(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=steps)

        if random.random() < 0.5:
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            y = y + noise_amp * np.random.normal(size=y.shape[0])

        if random.random() < 0.5:
            shift = np.random.randint(self.sr * self.duration)
            y = np.roll(y, shift)


        if random.random() < 0.5:
            y = np.flip(y)
        
        if random.random() < 0.5:
            y = self.apply_equalizer(y)
        
        return y

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def apply_equalizer(self, y):
        # Define frequency bands (in Hz)
        bands = [
            (20, 300),   # Bass
            (300, 2000), # Mid
            (2000, 8000) # Treble
        ]
       
        # Apply gain to each band
        for lowcut, highcut in bands:
            # Ensure the band frequencies are within the valid range
            if lowcut < self.sr / 2 and highcut < self.sr / 2:
                gain = np.random.uniform(0.5, 1.5) # Random gain between 0.5 and 1.5
                y = self.bandpass_filter(y, lowcut, highcut, self.sr) * gain
            
        return y

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super(LearnedPositionalEncoding, self).__init__()
        # Learnable positional encodings
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)  # Initialize with small random values

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]



def train(model, train_loader, criterion, optimizer, device, epoch, desired_batch_size):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels, _) in enumerate(train_loader):
        if inputs.size(0) != desired_batch_size:
            continue
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()
        average_loss = running_loss / (i + 1)
        accuracy = 100 * correct / total

        print(f"Epoch: {epoch + 1}, {((i + 1) / len(train_loader)) * 100:.0f}% complete, "
              f"Loss: {loss:.4f}, Avg Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")


def test(model, test_loader, criterion, device, log_name):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            all_targets.extend(target.cpu().numpy())
            all_outputs.extend(predicted.cpu().numpy())

    test_loss /= total
    accuracy = 100 * correct / total

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)")

    # Log the results
    result_string = f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)\n"
    with open(log_name, "a+") as file:
        file.write(result_string)
def validate(model, validation_loader, criterion, device, log_name, incorrect_log, epoch):
    model.eval()
    validation_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for data, target, file_names in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            validation_loss += loss.item()  # Sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct_preds = pred.eq(target.view_as(pred))
            correct += correct_preds.sum().item()
            total += data.size(0)

            all_targets.extend(target.view_as(pred).cpu().numpy())
            all_outputs.extend(pred.cpu().numpy())

            # Log incorrect file names
            incorrect_preds = ~correct_preds.view(-1)
            incorrect_files = [file_names[i] for i in range(len(file_names)) if incorrect_preds[i]]
            with open(incorrect_log, 'a+') as file:
                for file_name in incorrect_files:
                    file.write(file_name + '\n')

    validation_loss /= total
    accuracy = 100 * correct / total

    print(f"Validation set, Epoch {epoch + 1}: Average loss: {validation_loss:.4f}, "
          f"Accuracy: {correct}/{total} ({accuracy:.0f}%)")

    # Log results
    result_string = f"Validation {epoch + 1}: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)\n"
    with open(log_name, "a+") as file:
        file.write(result_string)

    return validation_loss

def run_vit(save_path, log_name, lr, bs, val_log_name, incorrect_v_log, patch_size=16, num_heads=8, num_layers=8, embedding_dim=512, num_classes=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    # Initialize the Vision Transformer model (removed num_patches argument)
    model = VisionTransformer(patch_size, embedding_dim, num_heads, num_layers, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs = 1000

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

    # Data directories
    holdout_ai_directory = './data/validation_set/ai_split'
    holdout_human_directory = './data/validation_set/human_split'
    ai_directory = './data/ai_split'
    human_directory = './data/human_split'



    # Seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Datasets
    dataset = AudioDatasetForViT(ai_directory, human_directory)
    validation_dataset = AudioDatasetForViT(holdout_ai_directory, holdout_human_directory)

    # Splitting dataset into train and test
    total_size = len(dataset)
    test_size = int(total_size * 0.2)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=cpu_count())
    validation_loader = DataLoader(validation_dataset, batch_size=bs, shuffle=True, num_workers=cpu_count())

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training and testing loop
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, bs)
        test(model, test_loader, criterion, device, log_name)

        # Validation phase
        val_loss = validate(model, validation_loader, criterion, device, val_log_name, incorrect_log=incorrect_v_log, epoch=epoch)

        scheduler.step(val_loss)

        # Save model after each epoch
        torch.save(model.state_dict(), f'{save_path}_{epoch}')

    # Save final model
    torch.save(model.state_dict(), f'{save_path}_final_training')


if __name__ == '__main__':
    run_vit(
        save_path='./data/models/training_models_vit/vit.pth', 
        log_name='./Vit_Logs/vision_transformer_log.txt', 
        lr=0.00001, 
        bs=16, 
        val_log_name='./Vit_Logs/vision_transformer_val_log.txt', 
        incorrect_v_log='./Vit_Logs/vision_transformer_incorrect.txt'
    )
