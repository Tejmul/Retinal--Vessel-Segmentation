"""Network architecture for Study Group Learning (SGL) retinal vessel segmentation.

Faithful reproduction of the two-stage model ("CON") from:
    Zhou, Yu, Shi. "Study Group Learning: Improving Retinal Vessel
    Segmentation Trained with Noisy Labels." MICCAI 2021.
    (official code: SHI-Labs/SGL-Retinal-Vessel-Segmentation, model/con.py)

The model is a cascade of two U-Net style networks:
    * EnhanceNet : RGB image (0..255) -> 3-channel enhancement map (0..1)
    * SegNet     : enhancement map    -> 1-channel vessel probability (0..1)

Only the final segmentation output is supervised; the enhancement map is
learned implicitly through the segmentation loss.
"""
import torch
import torch.nn as nn


def _conv_block(channel_in, channel_out):
    """Pre-activation conv block matching the official implementation.

    For the very first block (RGB input, channel_in == 3) there is no leading
    PReLU/BatchNorm; every deeper block is pre-activated (PReLU + BN before the
    convolution).
    """
    if channel_in == 3:
        return nn.Sequential(
            nn.Conv2d(channel_in, channel_out, 3, 1, 1),
            nn.PReLU(),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )
    return nn.Sequential(
        nn.PReLU(),
        nn.BatchNorm2d(channel_in),
        nn.Conv2d(channel_in, channel_out, 3, 1, 1),
        nn.PReLU(),
        nn.BatchNorm2d(channel_out),
        nn.Conv2d(channel_out, channel_out, 3, 1, 1),
    )


def _upconv(channel_in, channel_out):
    return nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)


class _UNet(nn.Module):
    """Shared 5-level U-Net body used by both EnhanceNet and SegNet.

    Channels: 3 -> 32 -> 64 -> 128 -> 256 -> 512 (encoder), symmetric decoder
    with skip connections. The final 1x1-ish block consumes the decoder feature
    (32ch) concatenated with the raw input (``in_ch``) -> ``out_ch``.
    """

    def __init__(self, in_ch, out_ch, scale_input):
        super().__init__()
        self.scale_input = scale_input  # EnhanceNet divides RGB by 255
        self.conv1 = _conv_block(3, 32)
        self.conv2 = _conv_block(32, 64)
        self.conv3 = _conv_block(64, 128)
        self.conv4 = _conv_block(128, 128 * 2)
        self.conv5 = _conv_block(128 * 2, 128 * 4)
        self.pool = nn.MaxPool2d(2)
        self.upconv1 = _upconv(64, 32)
        self.upconv2 = _upconv(128, 64)
        self.upconv3 = _upconv(128 * 2, 128)
        self.upconv4 = _upconv(128 * 4, 128 * 2)
        self.conv6 = _conv_block(128 * 4, 128 * 2)
        self.conv7 = _conv_block(128 * 2, 128)
        self.conv8 = _conv_block(128, 64)
        self.conv9 = _conv_block(64, 32)
        self.conv11 = _conv_block(32 + in_ch, out_ch)

    def forward(self, x):
        if self.scale_input:
            x = x / 255.0
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))
        u4 = self.conv6(torch.cat([self.upconv4(x5), x4], 1))
        u3 = self.conv7(torch.cat([self.upconv3(u4), x3], 1))
        u2 = self.conv8(torch.cat([self.upconv2(u3), x2], 1))
        u1 = self.conv9(torch.cat([self.upconv1(u2), x1], 1))
        u1 = torch.cat([u1, x], 1)
        return torch.sigmoid(self.conv11(u1))


class EnhanceNet(_UNet):
    def __init__(self):
        super().__init__(in_ch=3, out_ch=3, scale_input=True)


class SegNet(_UNet):
    def __init__(self):
        super().__init__(in_ch=3, out_ch=1, scale_input=False)


class CON(nn.Module):
    """Enhancement + Segmentation cascade (official ``MainNet``)."""

    def __init__(self):
        super().__init__()
        self.s1 = EnhanceNet()
        self.s2 = SegNet()

    def forward(self, x):
        enhance = self.s1(x)
        seg = self.s2(enhance)
        return enhance, seg


def build_model():
    return CON()


if __name__ == "__main__":
    net = build_model()
    n = sum(p.numel() for p in net.parameters())
    print(f"CON parameters: {n/1e6:.2f}M")
    dummy = torch.rand(1, 3, 64, 64) * 255
    enh, seg = net(dummy)
    print("enhance:", tuple(enh.shape), "seg:", tuple(seg.shape))
