import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, kernel_size=15):
        super(BasicBlock1D, self).__init__()
        # First convolution
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    """
    ResNet-1D architecture for ECG signal classification.
    Adapted from standard ResNet but with 1D convolutions and larger kernels
    to capture temporal dynamics of ECG.
    """
    def __init__(self, block, layers, num_leads=12, num_classes=5, input_len=2500, kernel_size=15):
        super(ResNet1D, self).__init__()
        self.inplanes = 64
        
        # Initial convolution
        # Stride 2 to reduce length early
        self.conv1 = nn.Conv1d(
            num_leads, 64, kernel_size=kernel_size, stride=2, 
            padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(block, 64, layers[0], kernel_size=kernel_size)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, kernel_size=kernel_size)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, kernel_size=kernel_size)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, kernel_size=kernel_size)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=15):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        """Returns spatial features before pooling."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet1d18(num_leads=12, num_classes=5, **kwargs):
    """Constructs a ResNet-1D-18 model."""
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_leads=num_leads, num_classes=num_classes, **kwargs)

def resnet1d34(num_leads=12, num_classes=5, **kwargs):
    """Constructs a ResNet-1D-34 model."""
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], num_leads=num_leads, num_classes=num_classes, **kwargs)
