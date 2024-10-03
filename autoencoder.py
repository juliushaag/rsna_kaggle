import json
import torch
from torchvision.models import resnet18, ResNet18_Weights

import torch.nn as nn
import torch.nn.functional as F


class ResNetAutoencoder(nn.Module):
  def __init__(self):
    super(ResNetAutoencoder, self).__init__()
    
    # Load a pre-trained ResNet18 model
    self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Adjust the first convolutional layer to accept single-channel images
    self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Remove the fully connected layer
    self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
    
    # Decoder layers with adjusted dimensions to output 512x512
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
      nn.ReLU(True),
      nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
      nn.Sigmoid()
    )
    
  def forward(self, x):
    x = self.encoder(x)
    print(x.shape)
    x = self.decoder(x)
    return x


# Example usage
if __name__ == "__main__":
  imgs = json.load(open("processed/images.json", "r"))

  img_data = torch.randn((1, 1, 512, 512)).float()
  print(img_data.shape)

  model = ResNetAutoencoder()

  output_image = model(img_data)
  F.mse_loss(img_data, output_image)
