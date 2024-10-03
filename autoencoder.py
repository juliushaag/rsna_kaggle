import json
import random
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt

class ResNet(nn.Module):
    def __init__(
        self,
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
    ) -> None:
        super().__init__()

        self.inplanes = 64
        self.dilation = 1
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.LayerNorm(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                nn.LayerNorm(planes * BasicBlock.expansion * 4),
            )

        layers = []
        layers.append(
            BasicBlock(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, nn.LayerNorm
            )
        )
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=nn.LayerNorm,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



class ResNetAutoencoder(nn.Module):
  def __init__(self):
    super(ResNetAutoencoder, self).__init__()
    
    self.embd_dim = 1024  
    # Load a pre-trained ResNet18 model
    # self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    self.encoder = ResNet(layers=[2, 2, 2, 2], num_classes=self.embd_dim)

    # Adjust the first convolutional layer to accept single-channel images
    self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Adjust the fully connected layer to output 512x16x16

    
    
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
    self.imgs = list(json.load(open("processed/lookup.json", "r")).values())[:1]
    
  def __len__(self):
    return len(self.imgs)
  
  def __getitem__(self, idx):
    img = np.load(self.imgs[idx] + ".npy")
    return (torch.tensor(img).unsqueeze(0) / 65504.0)

dataset = torch.utils.data.dataloader.DataLoader(RSNADataSet(), batch_size=BS, shuffle=True, num_workers=12, prefetch_factor=8)

random.seed(42)
# Example usage
if __name__ == "__main__":
  model = ResNetAutoencoder().to(DEVICE)
  opt = torch.optim.RMSprop(model.parameters(), lr=1e-4)

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
      loss = F.mse_loss(y, img)
      loss.backward()
      norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
      opt.step()
      losses.append(loss.item())
      epoch_loss += loss.item()
      epoch_norm += norm
      # print(f"Epoch {i:4} | Loss: {loss:.4f} | Norm: {norm:.4f}")



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
  