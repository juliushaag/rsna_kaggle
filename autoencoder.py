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
    self.encoder = resnet18(weights=True)
    
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
    # Adjust the first convolutional layer to accept single-channel images
    x = torch.cat([x, x, x], dim=2)  # Convert single-channel to 3-channel by duplicating the channel
    x = self.encoder(x)
    x = x.view(x.size(0), 512, 1, 1)  # Reshape for the decoder
    x = self.decoder(x)
    return x




# Example usage
if __name__ == "__main__":

  imgs = json.load(open("images.json", "r"))


  img_data = torch.tensor(pydicom.dcmread(list(imgs.values())[0]).pixel_array)
  print(img_data.shape)

  model = ResNetAutoencoder()

  input_image = torch.randn(1, 3, 512, 512)  # Example input
  output_image = model(img_data.unsqueeze(0))