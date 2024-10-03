import json
import random
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

class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super(ResNetAutoencoder, self).__init__()

        self.embd_dim = 1024  
        # Load a pre-trained ResNet18 model
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adjust the first convolutional layer to accept single-channel images
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adjust the fully connected layer to output 512x16x16
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, self.embd_dim)

        self.l1 = nn.Linear(self.embd_dim, self.embd_dim)
        self.ln1 = nn.LayerNorm(self.embd_dim)

        self.l2 = nn.Linear(self.embd_dim, self.embd_dim)
        self.ln2 = nn.LayerNorm(self.embd_dim)

        self.l3 = nn.Linear(self.embd_dim, self.embd_dim)
        self.ln3 = nn.LayerNorm(self.embd_dim)

        self.l4 = nn.Linear(self.embd_dim, self.embd_dim)
        self.ln4 = nn.LayerNorm(self.embd_dim)

        self.proj = nn.Linear(self.embd_dim, 512 * 16 * 16)

        # Decoder layers with adjusted dimensions to output 512x512 images
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        B, *_ = x.shape
        x = self.encoder(x)

        x = x + F.relu(self.ln1(self.l1(x)))
        x = x + F.relu(self.ln2(self.l2(x)))
        x = x + F.relu(self.ln3(self.l3(x)))
        x = x + F.relu(self.ln4(self.l4(x)))

        x = self.proj(x).view(B, 512, 16, 16)
        x = self.decoder(x)
        return (F.sigmoid(x) + 1) / 2


BS = 24
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RSNADataSet(torch.utils.data.Dataset):
    def __init__(self):
        self.imgs = list(json.load(open("processed/lookup.json", "r")).values())[:24]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = np.load(self.imgs[idx] + ".npy")
        return (torch.tensor(img).unsqueeze(0) / 65504.0 + 1) / 2

dataset = torch.utils.data.dataloader.DataLoader(RSNADataSet(), batch_size=BS, shuffle=True, num_workers=12, prefetch_factor=8)

random.seed(42)
# Example usage
if __name__ == "__main__":
    model = ResNetAutoencoder().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.grad_scaler.GradScaler()

    print("Training model with", sum(p.element_size() * p.nelement() for p in model.parameters()) // (2**20), "MB parameters")

    losses = list()
    norm = 0.0
    for i in range(EPOCHS):
        epoch_loss = 0
        epoch_norm = 0
        for img in tqdm(dataset):

            img = img.to(DEVICE)

            model.zero_grad()
            y = model(img)
            assert (img > 0).all()
            loss = F.binary_cross_entropy(y, img)
            
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            epoch_loss += loss.item()
            epoch_norm += norm

    plt.plot(np.log(losses))
    plt.show()

    img = next(iter(dataset)).to(DEVICE)
    y = model(img)

    for k in range(len(y)):
        plt.subplot(1, 2, 1)
        plt.imshow(img[k].cpu().numpy().squeeze(), cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(y[k].detach().cpu().numpy().squeeze(), cmap="gray")
        plt.show()
