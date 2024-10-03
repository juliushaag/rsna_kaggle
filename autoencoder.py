import json
import torch
from torchvision.models import resnet18

import torch.nn as nn
import torch.nn.functional as F

import pydicom


class ResNetAutoencoder(nn.Module):
  def __init__(self):
    super(ResNetAutoencoder, self).__init__()
    
    # Load a pre-trained ResNet18 model
    self.encoder = resnet18(pretrained=True)
    
    # Remove the fully connected layer
    self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
    
    # Decoder layers
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.ReLU(True),
      nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.Sigmoid()
    )
    
  def forward(self, x):
    x = self.encoder(x)
    x = x.view(x.size(0), 512, 1, 1)  # Reshape for the decoder
    x = self.decoder(x)
    return x




# Example usage
if __name__ == "__main__":

  imgs = json.loads("images.json")

  img = imgs.values[0]

  model = ResNetAutoencoder()
  print(model)
  input_image = torch.randn(1, 3, 224, 224)  # Example input
  output_image = model(input_image)
  print(output_image.shape)