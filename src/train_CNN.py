import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import glob
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, f1_score, confusion_matrix
from models.cnn_model import CNNTest
from multiprocessing import cpu_count


# Augmentation functions
def augment_audio(y, sr):
    # Pitch Shift
    if np.random.rand() < 0.5:
        steps = np.random.randint(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    
    # Additive Noise
    if np.random.rand() < 0.5:
        noise = np.random.randn(len(y))
        y = y + 0.005 * noise

    # Time Shifting
    if np.random.rand() < 0.5:
        shift = np.random.randint(-int(0.1 * sr), int(0.1 * sr))
        y = np.roll(y, shift)
    
    return y

def spec_augment(spec):
    num_mask = 2
    freq_masking_max_percentage = 0.15
    time_masking_max_percentage = 0.3
    for i in range(num_mask):
        all_freqs_num = spec.shape[0]
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        spec[int(f):int(f) + num_freqs_to_mask, :] = 0

        all_frames_num = spec.shape[1]
        time_percentage = np.random.uniform(low=0.0, high=time_masking_max_percentage)
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        spec[:, int(t):int(t) + num_frames_to_mask] = 0
    return spec

class AudioDataset(Dataset):
    def __init__(self, ai_directory, human_directory, sr=16000, duration=3, augment=True):
        self.ai_files = glob.glob(os.path.join(ai_directory, '*.mp3'))
        self.human_files = glob.glob(os.path.join(human_directory, '*.mp3'))
        self.all_files = self.ai_files + self.human_files
        self.labels = [0] * len(self.ai_files) + [1] * len(self.human_files)
        self.sr = sr
        self.duration = duration
        self.augment = augment
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
                    if random.random() < 0.20:
                        y = augment_audio(y, sr)

                y = np.clip(y, -1.0, 1.0)
                S = np.abs(librosa.stft(y))**2
                S_db = librosa.power_to_db(S + 1e-10, ref=np.max)
                S_db = (S_db - self.global_mean) / self.global_std

                target_shape = (1025, 94)
                if S_db.shape != target_shape:
                    S_db = np.pad(S_db, (
                        (0, max(0, target_shape[0] - S_db.shape[0])), 
                        (0, max(0, target_shape[1] - S_db.shape[1]))
                    ), mode='constant', constant_values=-self.global_mean)
                    S_db = S_db[:target_shape[0], :target_shape[1]]

                if self.augment and random.random() < 0.2:
                    S_db = spec_augment(S_db)
                
                spectrogram_tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)
                return spectrogram_tensor, label, audio_path
            except Exception as e:
                print(f"Skipping file {audio_path} due to error: {e}")
                idx = (idx + 1) % len(self.all_files)


def train(model, train_loader, criterion, optimizer, device, epoch, desired_batch_size):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels, file_names) in enumerate(train_loader, 0):
        if inputs.size(0) != desired_batch_size:
            continue
        try:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()
            average_loss = running_loss / (i + 1)
            accuracy = 100 * correct / total

            result_string = f'{((i + 1) / len(train_loader)) * 100:.0f}%, Epoch: {epoch}/{15}, Sample:{i}/{len(train_loader)}, Loss: {loss:.4f}, Avg Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%'
            print(result_string)
        except Exception as e:
            print(f"Error encountered in batch {i}: {e}")
            continue

def test(model, test_loader, criterion, device, log_name):
    model.eval()
    test_loss = 0
    correct = 0
    valid_samples = 0
    all_targets = []
    all_outputs = []
    all_probabilities = []

    with torch.no_grad():
        for data, target, file_names in test_loader:
            if data is None:
                continue

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            probabilities = F.softmax(output, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
            pred = output.argmax(dim=1, keepdim=True)

            correct_preds = pred.eq(target.view_as(pred))
            correct += correct_preds.sum().item()
            valid_samples += data.size(0)

            all_targets.extend(target.view_as(pred).cpu().numpy())
            all_outputs.extend(pred.cpu().numpy())

            incorrect_preds = ~correct_preds.view(-1)
            incorrect_files = [file_names[i] for i in range(len(file_names)) if incorrect_preds[i]]
            os.system('touch ./CNN_Logs/incorrectfiles.txt')
            with open("./CNN_Logs/incorrectfiles.txt", 'a+') as file:
                for file_name in incorrect_files:
                    file.write(file_name + '\n')

    test_loss /= valid_samples
    accuracy = 100 * correct / valid_samples

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_outputs, average='binary')
    roc_auc = roc_auc_score(all_targets, [prob[1] for prob in all_probabilities])

    mcc = matthews_corrcoef(all_targets, all_outputs)
    f1 = f1_score(all_targets, all_outputs, average='weighted')
    cm = confusion_matrix(all_targets, all_outputs)

    avg_log_loss = test_loss

    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{valid_samples} ({accuracy:.0f}%), "
          f"MCC: {mcc:.4f}, F1: {f1:.4f}, "
          f"Average Log Loss: {avg_log_loss:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    result_string = f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{valid_samples} ({accuracy:.0f}%), MCC: {mcc}, F1: {f1}, Average Log Loss: {avg_log_loss}, Confusion Matrix: {cm}, Precision:{precision}, Recall: {recall}, ROC AUC: {roc_auc}"
    with open(log_name, "a+") as file:
        file.write(f'{result_string}\n')

def validate(model, validation_loader, criterion, device, log_name, incorrect_log, epoch):
    model.eval()
    validation_loss = 0
    correct = 0
    valid_samples = 0
    all_targets = []
    all_outputs = []
    total_log_loss = 0

    with torch.no_grad():
        for data, target, file_names in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            validation_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            correct_preds = pred.eq(target.view_as(pred))
            correct += correct_preds.sum().item()
            valid_samples += data.size(0)

            all_targets.extend(target.view_as(pred).cpu().numpy())
            all_outputs.extend(pred.cpu().numpy())

            incorrect_preds = ~correct_preds.view(-1)
            incorrect_files = [file_names[i] for i in range(len(file_names)) if incorrect_preds[i]]
            with open(incorrect_log, 'a+') as file:
                for file_name in incorrect_files:
                    file.write(file_name + '\n')

    validation_loss /= valid_samples
    accuracy = 100 * correct / valid_samples

    mcc = matthews_corrcoef(all_targets, all_outputs)
    f1 = f1_score(all_targets, all_outputs, average='weighted')
    cm = confusion_matrix(all_targets, all_outputs)

    avg_log_loss = total_log_loss / len(validation_loader)

    print(f'Validation set: Average loss: {validation_loss:.4f}, '
          f'Accuracy: {correct}/{valid_samples} ({accuracy:.0f}%), '
          f'MCC: {mcc:.4f}, F1: {f1:.4f}, '
          f'Average Log Loss: {avg_log_loss:.4f}')
    print(f'Confusion Matrix:\n{cm}')

    result_string = f'\nValidation {epoch}: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{valid_samples} ({accuracy:.0f}%), MCC: {mcc}, F1: {f1}, Average Log Loss: {avg_log_loss}, Confusion Matrix: {cm}'
    with open(log_name, "a+") as file:
        file.write(f'{result_string}\n')
    return validation_loss

def run(save_path, log_name, lr, bs, val_log_name, incorrect_v_log):
    model = CNNTest()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs = 50
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

    holdout_ai_directory = './data/validation_set/ai_split'
    holdout_human_directory = './data/validation_set/human_split'
    ai_directory = './data/ai_split'
    human_directory = './data/human_split'

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    dataset = AudioDataset(ai_directory, human_directory)
    validation_dataset = AudioDataset(holdout_ai_directory, holdout_human_directory)

    total_size = len(dataset)
    test_size = int(total_size * 0.2)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=cpu_count())
    validation_loader = DataLoader(validation_dataset, batch_size=bs, shuffle=True, num_workers=cpu_count())

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, bs)
        test(model, test_loader, criterion, device, log_name)

        val_loss = validate(model, validation_loader, criterion, device, val_log_name, incorrect_log=incorrect_v_log, epoch=epoch)

        scheduler.step(val_loss)
        torch.save(model.state_dict(), f'{save_path}_{epoch}')

    torch.save(model.state_dict(), f'{save_path}_final_training')

if __name__ == "__main__":
    os.system('touch ./CNN_Logs/regularlog1.txt')
    os.system('touch ./CNN_Logs/regular_validation1.txt')
    os.system('touch ./CNN_Logs/regular_incorrect_validation1.txt')
    path1 = 'models/Your_CNN_Model.pth'
    log1 = './CNN_Logs/regularlog1.txt'
    vlog1 = './CNN_Logs/regular_validation1.txt'
    ivlog1 = './CNN_Logs/regular_incorrect_validation1.txt'

    run(
        save_path=path1,
        log_name=log1,
        lr=0.00008,
        bs=16,
        val_log_name=vlog1,
        incorrect_v_log=ivlog1
    )
