import torch
import torch.nn as nn

import torchvision
from torchvision.models import resnet18


class ResNet4ch(nn.Module):
    def __init__(self, pretrained=False, use_mask_channel=True):
        super().__init__()
        self.use_mask_channel = use_mask_channel
        self.backbone = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if use_mask_channel:
            old = self.backbone.conv1
            new = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=old.bias)
            with torch.no_grad():
                new.weight[:, :3] = old.weight
                new.weight[:, 3:] = 0.0
            self.backbone.conv1 = new
        self.fc = nn.Linear(self.backbone.fc.in_features, 1)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        feat = self.backbone(x)
        logit = self.fc(feat)
        return logit.squeeze(1)

    @classmethod
    def from_pretrained(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model = cls(use_mask_channel=True)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        return model
