import json
import random
from typing import Optional, Tuple
import numpy as np
import torch
import torch.amp
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torchvision.models import resnet18, ResNet18_Weights

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt


class ResidualBlock(nn.Module):
    """Residual block for the decoder to mirror ResNet architecture"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.silu(out)
        
        return out


class DecoderBlock(nn.Module):
    """Enhanced decoder block with residual connections"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=stride,
            padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.residual = ResidualBlock(out_channels, out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.residual(x)
        return x


class ResNetAutoencoder(nn.Module):
    def __init__(self, 
                 emb_dim: int = 2056,
                 input_channels: int = 1,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.emb_dim = emb_dim
        
        # Encoder (modified ResNet-18)
        self.encoder = resnet18()
        
        # Modify first layer for different input channels
        self.encoder.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )    
    
        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout_rate),
            nn.GELU()
        )
    
        # Projection from embedding to spatial features
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, 256 * 16 * 16),
            nn.LayerNorm(256 * 16 * 16),
            nn.Dropout(dropout_rate),
            nn.GELU()
        )
        
        # Enhanced decoder with residual connections
        # Decoder with enhanced residuals
        self.decoder = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 16),
            nn.ConvTranspose2d(16, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights using kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        
        return self.encoder.fc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to image space"""
        x = self.proj(z)
        x = x.view(-1, 256, 16, 16)
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional variational outputs"""
        z = self.encode(x)
        return self.decode(z)

    def get_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate loss"""
        recon_x = self(x)
        loss = F.binary_cross_entropy(recon_x, x)
        return loss
        



class RSNADataSet(torch.utils.data.Dataset):
    def __init__(self, index = 0, max_elements = 1000000):
        self.imgs = torch.load(f"data/img_shard_{index}.pt", weights_only=True)[:max_elements].to(torch.float32)
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]
     


# Example usage
if __name__ == "__main__":

    random.seed(42)

    BS = 16
    EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model = ResNetAutoencoder(2056 * 2).to(DEVICE)
    
    model = torch.load("models/autoencoder.pt", weights_only=False).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    
    train_dataset = torch.utils.data.dataloader.DataLoader(
        RSNADataSet(2), batch_size=BS, shuffle=True, num_workers=1, pin_memory=True, prefetch_factor=2, persistent_workers=True, timeout=60)
    test_dataset = torch.utils.data.dataloader.DataLoader(RSNADataSet(1, 1000), batch_size=BS, pin_memory=True)
    

    print(f"Training model with {sum(p.element_size() * p.nelement() for p in model.parameters()) / (2**30):.3f}GB parameters")

    train_losses = list()
    test_losses = list()
    train_norms = list()
    norm = 0.0

    test_intervall = 200

    model.train()
    for n_epoch in range(EPOCHS):
        intervall_loss = list()
        intervall_norm = list()
        for i, batch in tqdm(enumerate(train_dataset)):
            batch = batch.to(DEVICE)
            
            opt.zero_grad()
            loss = model.get_loss(batch)
            
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            intervall_loss.append(loss.item())
            intervall_norm.append(norm.item())
            
            if i % test_intervall == 0 and i != 0:
                model.eval()
                with torch.no_grad():
                    losses = list()
                    for batch in test_dataset:
                        batch = batch.to(DEVICE)
                        loss = model.get_loss(batch)
                        losses.append(loss.item())

                    test_loss = sum(losses) / len(losses)
                    
                    train_loss = sum(intervall_loss) / len(intervall_loss)
                    intervall_loss.clear()

                    train_norm = sum(intervall_norm) / len(intervall_norm)
                    intervall_norm.clear()


                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    train_norms.append(train_norm)
                    print(f"Epoch {n_epoch}:{i} train_loss: {train_loss:.4f} test_loss: {test_loss:.4f} norm: {train_norm:.4f}")
                model.train()
     
    torch.save(model, "models/autoencoder.pt")

    plt.plot(np.log(train_losses), label="train loss")
    plt.plot(np.log(test_losses), label="test loss")
    plt.legend()
    plt.show()

    model.eval()
    imgs = next(iter(test_dataset))[:4].to(DEVICE)
    
    del test_dataset
    del train_dataset

    with torch.no_grad():
        pred = model(imgs)

    plt.figure(figsize=(16, 8))
    for i in range(len(imgs)):

        v_img = imgs[i].cpu().numpy().squeeze()
        v_pred = pred[i].detach().cpu().numpy().squeeze()
        plt.subplot(2, 4, i * 2 + 1)
        plt.imshow(v_img, cmap="gray")
        plt.subplot(2, 4, i * 2 + 2)
        plt.imshow(v_pred, cmap="gray")
    
    plt.show()

