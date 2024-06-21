import torch.nn as nn
import torch


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super(LearnedPositionalEncoding, self).__init__()
        # Learnable positional encodings
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)  # Initialize with small random values

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]

class VisionTransformer(nn.Module):
    def __init__(self, patch_size, embedding_dim, num_heads, num_layers, num_classes, device):
        super(VisionTransformer, self).__init__()
        # Assume a specific fixed input image size based on the sample and bit rate per 3s clip.
        num_patches_h = 1025 // patch_size  
        num_patches_w = 94 // patch_size
        num_patches = num_patches_h * num_patches_w
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim, device=device))
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim).to(device)

        # Initialize linear projection layer
        self.linear_proj = nn.Linear(patch_size * patch_size, embedding_dim).to(device)
        self.positional_encoding = LearnedPositionalEncoding(num_patches, embedding_dim).to(device)

        # Transformer encoder layers
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=embedding_dim * 4, 
            batch_first=True, 
            dropout=0.1, 
            activation='gelu',
            layer_norm_eps=1e-6
        ).to(device)
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers).to(device)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            nn.Dropout(0.1),
        ).to(device)

    def forward(self, x):
        B, C, H, W = x.size()

        # Calculate the number of patches dynamically based on the input dimensions and prev init patch sizes
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # Split the input into patches
        x_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_patches = x_patches.contiguous().view(B, -1, self.patch_size * self.patch_size)

        # Apply linear projection
        x_proj = self.linear_proj(x_patches)

        # Add the classification token
        cls_token = self.cls_token.expand(B, -1, -1).to(x.device)
        x = torch.cat((cls_token, x_proj), dim=1)

        # Add positional encoding to the patches
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)
        
        # Classification head
        cls_output = x[:, 0]
        cls_output = self.layer_norm(cls_output)
        return self.classifier(cls_output)



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
