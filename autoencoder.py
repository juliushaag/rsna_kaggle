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

from tools.prep_imgs import transform_img


class ResNetAutoencoder(nn.Module):
    def __init__(self, emb_dim = 2056):
        super(ResNetAutoencoder, self).__init__()

        self.embd_dim = emb_dim

        # Load a pre-trained ResNet18 model
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adjust the first convolutional layer to accept single-channel images
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Adjust the fully connected layer to output 512x16x16
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, self.embd_dim)

        self.proj = nn.Linear(self.embd_dim, 256 * 256)

        # Decoder layers with adjusted dimensions to output 512x512 images
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.encoder(x)

    def autoencode(self, img):
        B, *_ = img.shape
        x = self(img)
        x = self.proj(x).view(B, 256, 16, 16)
        x = self.decoder(x)
        return x


BS = 32
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    # model = ResNetAutoencoder(2056 * 2)
    
    model = torch.load("models/autoencoder.pt", weights_only=False).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    train_dataset = torch.utils.data.dataloader.DataLoader(RSNADataSet(0), batch_size=BS, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=8)
    test_dataset = torch.utils.data.dataloader.DataLoader(RSNADataSet(1, 1000), batch_size=BS, pin_memory=True)
    

    print(f"Training model with {sum(p.element_size() * p.nelement() for p in model.parameters()) / (2**30):.3f}GB parameters")

    train_losses = list()
    test_losses = list()
    norm = 0.0

    test_intervall = 200

    model.train()
    for n_epoch in range(EPOCHS):
        intervall_loss = list()
        for i, batch in enumerate(train_dataset):
            batch = batch.to(DEVICE)
            model.zero_grad()

            y = model.autoencode(batch)
            loss = F.binary_cross_entropy(y, batch)

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            intervall_loss.append(loss.item())
            
            if i % test_intervall == 0 and i != 0:
                model.eval()
                with torch.no_grad():
                    losses = list()
                    for batch in test_dataset:
                        batch = batch.to(DEVICE)
                        y = model.autoencode(batch)
                        loss = F.binary_cross_entropy(y, batch)
                        losses.append(loss.item())
                    test_loss = sum(losses) / len(losses)
                    train_loss = sum(intervall_loss) / len(intervall_loss)
                    intervall_loss.clear()

                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    print(f"Epoch {n_epoch}:{i} train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}")
                model.train()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
     
    torch.save(model, "models/autoencoder.pt")

    plt.plot(np.log(train_losses))
    plt.plot(np.log(test_losses))
    plt.show()

    model.eval()
    imgs = next(iter(test_dataset))[:4].to(DEVICE)

    with torch.no_grad():
        pred = model.autoencode(imgs)

    plt.figure(figsize=(16, 8))
    for i in range(len(imgs)):

        v_img = imgs[i].cpu().numpy().squeeze()
        v_pred = pred[i].detach().cpu().numpy().squeeze()
        plt.subplot(2, 4, i * 2 + 1)
        plt.imshow(v_img, cmap="gray")
        plt.subplot(2, 4, i * 2 + 2)
        plt.imshow(v_pred, cmap="gray")
    
    plt.show()
