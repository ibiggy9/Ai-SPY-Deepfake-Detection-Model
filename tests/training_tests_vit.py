import unittest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.vit_model import VisionTransformer
from src.train_vision_transformer import AudioDatasetForViT

class TestAudioDatasetForViT(unittest.TestCase):

    def setUp(self):
        self.ai_directory = './data/ai_split'
        self.human_directory = './data/human_split'
        self.dataset = AudioDatasetForViT(self.ai_directory, self.human_directory, augment=False)
        self.augment_dataset = AudioDatasetForViT(self.ai_directory, self.human_directory, augment=True)

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

class TestVisionTransformer(unittest.TestCase):

    def setUp(self):
        self.patch_size = 16
        self.embedding_dim = 512
        self.num_heads = 8
        self.num_layers = 8
        self.num_classes = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.model = VisionTransformer(self.patch_size, self.embedding_dim, self.num_heads, self.num_layers, self.num_classes, self.device)

        self.dataset = AudioDatasetForViT('./data/ai_split', './data/human_split', augment=False)
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False)

    def test_model_forward(self):
        dataiter = iter(self.dataloader)
        spectrograms, labels, paths = dataiter.next()
        output = self.model(spectrograms.to(self.device))
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], self.num_classes)

    def test_training_step(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.to(self.device)
        for spectrograms, labels, paths in self.dataloader:
            spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            self.assertIsInstance(loss.item(), float)

if __name__ == '__main__':
    unittest.main()
