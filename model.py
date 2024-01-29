import torch
import torch.nn as nn
import torch.nn.functional as F


class GConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, activ, upsample=False,
    ):
        super().__init__()

        self.activ = activ

        if upsample:
            conv_layer = nn.ConvTranspose2d
        else:
            conv_layer = nn.Conv2d
        self.conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # "We use instance normalization for the generator."
        self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activ == "relu":
            x = torch.relu(x)
        elif self.activ == "tanh":
            x = torch.tanh(x)
        return x


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = GConvBlock(
            256, 256, kernel_size=3, stride=1, padding=1, activ="relu",
        )

    def forward(self, x):
        return x + self.conv(x)


class Generator(nn.Module):
    def __init__(self, n_domains):
        super().__init__()

        # "StarGAN has the generator network composed of two convolutional layers
        # with the stride size of two for downsampling, six residual blocks, and two
        # transposed convolutional layers with the stride size of two for upsampling."
        self.in_layer = GConvBlock(
            n_domains + 2, 64, kernel_size=7, stride=1, padding=3, activ="relu", upsample=False,
        )
        self.downsample1 = GConvBlock(
            64, 128, kernel_size=4, stride=2, padding=1, activ="relu", upsample=False,
        )
        self.downsample2 = GConvBlock(
            128, 256, kernel_size=4, stride=2, padding=1, activ="relu", upsample=False,
        )
        self.bottleneck = nn.Sequential(*[ResBlock() for _ in range(6)])
        self.upsample1 = GConvBlock(
            256, 128, kernel_size=4, stride=2, padding=1, activ="relu", upsample=True,
        )
        self.upsample2 = GConvBlock(
            128, 64, kernel_size=4, stride=2, padding=1, activ="relu", upsample=True,
        )
        self.out_layer = GConvBlock(
            64, 3, kernel_size=7, stride=1, padding=3, activ="tanh", upsample=True,
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.bottleneck(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.out_layer(x)
        return x


class DConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True,
        )

    def forward(self, x):
        x = self.conv(x)
        # "We use Leaky ReLU with a negative slope of 0.01."
        x = F.leaky_relu(x, negative_slope=0.01) 
        return x


class Discriminator(nn.Module):
    # "We use no normalization for the discriminator."
    def __init__(self, n_domains):
        super().__init__()

        self.conv_block1 = DConvBlock(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv_block2 = DConvBlock(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv_block3 = DConvBlock(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv_block4 = DConvBlock(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv_block5 = DConvBlock(512, 1024, kernel_size=4, stride=2, padding=1)
        self.conv_block6 = DConvBlock(1024, 2048, kernel_size=4, stride=2, padding=1)
        self.src_out = DConvBlock(2048, 1, kernel_size=3, stride=1, padding=1)
        self.cls_out = DConvBlock(2048, n_domains, kernel_size=2, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x_src = self.src_out(x)
        x_cls = self.cls_out(x)
        return x_src.mean(dim=(1, 2, 3)), x_cls # We leverage PatchGANs."


if __name__ == "__main__":
    img_size = 128
    n_domains = 12

    x = torch.randn(2, n_domains + 2, img_size, img_size)
    gen = Generator(n_domains=n_domains)
    out = gen(x)
    print(out.shape)
    
    x = torch.randn(2, 3, img_size, img_size)
    disc = Discriminator(n_domains=n_domains)
    src_out, cls_out = disc(x)
    print(src_out.shape, cls_out.shape)
