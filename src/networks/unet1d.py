import torch
import torch.nn as nn


class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=[32, 64, 128]):
        super(UNet1D, self).__init__()

        self.encoder1 = self._block(in_channels, channels[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder2 = self._block(channels[0], channels[1])
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = self._block(channels[1], channels[2])

        self.upconv2 = nn.ConvTranspose1d(channels[2], channels[1], kernel_size=2, stride=2)
        self.decoder2 = self._block((channels[1]) * 2, channels[1])

        self.upconv1 = nn.ConvTranspose1d(channels[1], channels[0], kernel_size=2, stride=2)
        self.decoder1 = self._block(channels[1], channels[0])

        self.conv = nn.Conv1d(in_channels=channels[0], out_channels=out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1).squeeze(1)


if __name__ == "__main__":
    x = torch.randn(4, 20)
    model = UNet1D()
    print(model(x).shape)
