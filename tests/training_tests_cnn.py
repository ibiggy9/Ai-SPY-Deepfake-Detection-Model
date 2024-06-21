import unittest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.cnn_model import CNNTest
from src.train_CNN import AudioDataset

class TestAudioDataset(unittest.TestCase):

    def setUp(self):
        self.ai_directory = './data/ai_split'
        self.human_directory = './data/human_split'
        self.dataset = AudioDataset(self.ai_directory, self.human_directory, augment=False)
        self.augment_dataset = AudioDataset(self.ai_directory, self.human_directory, augment=True)

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), len(self.dataset.ai_files) + len(self.dataset.human_files))

    def test_get_item(self):
        sample, label, path = self.dataset[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertIsInstance(path, str)

    def test_augmentation(self):
        sample, label, path = self.augment_dataset[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertIsInstance(path, str)

class TestCNNTest(unittest.TestCase):

    def setUp(self):
        self.model = CNNTest()
        self.dataset = AudioDataset('./data/ai_split', './data/human_split', augment=False)
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False)

    def test_model_forward(self):
        dataiter = iter(self.dataloader)
        spectrograms, labels, paths = dataiter.next()
        output = self.model(spectrograms)
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 2)

    def test_training_step(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        for spectrograms, labels, paths in self.dataloader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self.model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            self.assertIsInstance(loss.item(), float)

if __name__ == '__main__':
    unittest.main()
