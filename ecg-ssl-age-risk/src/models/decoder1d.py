import torch
import torch.nn as nn

class ConvTransposeBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, padding=7, output_padding=0):
        super(ConvTransposeBlock1D, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, stride=stride, 
            padding=padding, output_padding=output_padding, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Decoder1D(nn.Module):
    """
    Decoder to upsample latent features (512, ~) back to (12, 2500).
    Mirror of ResNet1D-18 structure roughly.
    """
    def __init__(self, latent_dim=512, output_channels=12, output_len=2500):
        super(Decoder1D, self).__init__()
        
        # We assume the encoder reduces 2500 -> ~79 (stride 32 total)
        # 2500 / 2^5 = 78.125 -> 79 steps roughly if padding aligns.
        # We will use Interpolation to force size if needed, but lets try to map strides.
        
        # Input: (B, 512, 79) - coming from avgpool'd or just before avgpool?
        # Ideally we take pre-pooling features. 
        # ResNet18 downsamples: /2 (stem) -> /1 (L1) -> /2 (L2) -> /2 (L3) -> /2 (L4) = /16?
        # Let's check ResNet1D structure:
        # Conv1 (stride 2) + MaxPool (stride 2) = /4
        # L1: /1
        # L2: /2
        # L3: /2
        # L4: /2
        # Total: /4 * /2 * /2 * /2 = /32. 
        # 2500 / 32 = 78.125. Feature map width is 79.
        
        self.layer4_dec = ConvTransposeBlock1D(512, 256, stride=2, padding=7, output_padding=1) # 79 -> 158
        self.layer3_dec = ConvTransposeBlock1D(256, 128, stride=2, padding=7, output_padding=1) # 158 -> 316
        self.layer2_dec = ConvTransposeBlock1D(128, 64, stride=2, padding=7, output_padding=1)  # 316 -> 632
        self.layer1_dec = ConvTransposeBlock1D(64, 64, stride=1, padding=7) # 632 -> 632
        
        # Upsample stem
        # Reverse MaxPool (approx with ConvTranspose stride 2)
        self.up_pool = ConvTransposeBlock1D(64, 64, stride=2, padding=1, output_padding=0, kernel_size=3) # 632 -> 1264 roughly
        # Reverse Conv1 (stride 2)
        self.up_conv1 = nn.ConvTranspose1d(64, output_channels, kernel_size=15, stride=2, padding=7, output_padding=1) # 1264 -> ~2528
        
        self.output_len = output_len

    def forward(self, x):
        # x: (Batch, 512, 1) if from global pool, OR (Batch, 512, 79) if before pool.
        # We need spatial features for good reconstruction.
        # If x is (Batch, 512), we unflatten to (Batch, 512, 1) and upsample massive amount or repeat?
        # Better: SSL usually uses features BEFORE Global Avg Pool.
        
        x = self.layer4_dec(x)
        x = self.layer3_dec(x)
        x = self.layer2_dec(x)
        x = self.layer1_dec(x)
        
        x = self.up_pool(x)
        x = self.up_conv1(x)
        
        # Crop or Interpolate to exact length 2500
        if x.shape[2] != self.output_len:
            x = torch.nn.functional.interpolate(x, size=self.output_len, mode='linear', align_corners=False)
            
        return x
