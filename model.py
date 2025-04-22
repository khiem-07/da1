# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinySegNet(nn.Module):
    def __init__(self, n_classes):
        super(TinySegNet, self).__init__()
        # Encoder: giảm dần số kênh
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # [B, 16, 256, 256]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 256, 256]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # [B, 32, 128, 128]

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # [B, 64, 64, 64]

        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # [B, 64, 128, 128]
        # Kết hợp với output của enc3 (64 kênh)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),  # [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # [B, 32, 256, 256]
        # Kết hợp với output của enc2 (32 kênh)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 32, 16, kernel_size=3, padding=1),  # [B, 16, 256, 256]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Lớp output: chuyển từ 16 kênh về số lớp (3)
        self.out_conv = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)       # [B, 16, 256, 256]
        x2 = self.enc2(x1)      # [B, 32, 256, 256]
        x2p = self.pool1(x2)    # [B, 32, 128, 128]
        
        x3 = self.enc3(x2p)     # [B, 64, 128, 128]
        x3p = self.pool2(x3)    # [B, 64, 64, 64]
        
        x4 = self.enc4(x3p)     # [B, 64, 64, 64]
        
        # Decoder
        x = self.up1(x4)        # [B, 64, 128, 128]
        # Skip connection với x3
        x = torch.cat([x, x3], dim=1)  # [B, 64+64, 128, 128]
        x = self.dec1(x)        # [B, 32, 128, 128]
        
        x = self.up2(x)         # [B, 32, 256, 256]
        # Skip connection với x2
        x = torch.cat([x, x2], dim=1)  # [B, 32+32, 256, 256]
        x = self.dec2(x)        # [B, 16, 256, 256]
        
        x = self.out_conv(x)    # [B, n_classes, 256, 256]
        # Với CrossEntropyLoss, không cần softmax tại đầu ra.
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test mô hình với input 256x256
    model = TinySegNet(n_classes=3)
    print("Total parameters:", count_parameters(model))
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)
