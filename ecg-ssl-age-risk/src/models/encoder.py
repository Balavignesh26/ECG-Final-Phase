import torch
import torch.nn as nn

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=7, dropout=0.2):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet1d(nn.Module):
    """
    ResNet1d for ECG signal processing.
    Adapted from standard ResNet architectures for 1D data.
    """
    def __init__(self, input_channels=12, layers=[2, 2, 2, 2], num_filters=[64, 128, 256, 512], kernel_size=7, dropout=0.2):
        super(ResNet1d, self).__init__()
        self.inplanes = num_filters[0]
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, num_filters[0], kernel_size=kernel_size, stride=2, padding=kernel_size//2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(BasicBlock1d, num_filters[0], layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, num_filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, num_filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, num_filters[3], layers[3], stride=2)

        # Global Average Pooling to get fixed feature vector
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Output feature dimension
        self.output_dim = num_filters[3] * BasicBlock1d.expansion

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.kernel_size, self.dropout))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=self.kernel_size, dropout=self.dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
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
        
        return x

if __name__ == "__main__":
    # Test
    model = ResNet1d(input_channels=12)
    dummy_input = torch.randn(2, 12, 1000) # (Batch, Leads, Time)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
