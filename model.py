import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activ,
        upsample=False,
        normalize=True,
    ):
        super().__init__()

        self.activ = activ
        self.normalize = normalize

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
            bias=False if normalize else True,
        )
        if normalize:
            # "We use instance normalization for the generator."
            self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)

        if self.activ == "relu":
            x = torch.relu(x)
        elif self.activ == "tanh":
            x = torch.tanh(x)
        elif self.activ == "leaky_relu":
            # "We use Leaky ReLU with a negative slope of 0.01."
            x = F.leaky_relu(x, negative_slope=0.01)
        return x


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = ConvBlock(
            256, 256, kernel_size=3, stride=1, padding=1, activ="relu",
        )

    def forward(self, x):
        return x + self.conv(x)


class Generator(nn.Module):
    def __init__(self, dom_dim):
        super().__init__()

        # "StarGAN has the generator network composed of two convolutional layers
        # with the stride size of two for downsampling, six residual blocks, and two
        # transposed convolutional layers with the stride size of two for upsampling."
        self.in_layer = ConvBlock(
            dom_dim + 3,
            64,
            kernel_size=7,
            stride=1,
            padding=3,
            activ="relu",
            upsample=False,
            normalize=True,
        )
        self.downsample1 = ConvBlock(
            64,
            128,
            kernel_size=4,
            stride=2,
            padding=1,
            activ="relu",
            upsample=False,
            normalize=True,
        )
        self.downsample2 = ConvBlock(
            128,
            256,
            kernel_size=4,
            stride=2,
            padding=1,
            activ="relu",
            upsample=False,
            normalize=True,
        )
        self.bottleneck = nn.Sequential(*[ResBlock() for _ in range(6)])
        self.upsample1 = ConvBlock(
            256,
            128,
            kernel_size=4,
            stride=2,
            padding=1,
            activ="relu",
            upsample=True,
            normalize=True,
        )
        self.upsample2 = ConvBlock(
            128,
            64,
            kernel_size=4,
            stride=2,
            padding=1,
            activ="relu",
            upsample=True,
            normalize=True,
        )
        self.out_layer = ConvBlock(
            64,
            3,
            kernel_size=7,
            stride=1,
            padding=3,
            activ="tanh",
            upsample=True,
            normalize=True,
        )

    def forward(self, src_image, trg_dom):
        _, _, h, w = src_image.shape
        x = torch.cat([src_image, trg_dom[..., None, None].repeat(1, 1, h, w)], dim=1)
        x = self.in_layer(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.bottleneck(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.out_layer(x)
        return x


class Discriminator(nn.Module):
    """
    "We use no normalization for the discriminator."
    """
    def __init__(self, dom_dim):
        super().__init__()

        self.conv_block1 = ConvBlock(
            3,
            64,
            kernel_size=4,
            stride=2,
            padding=1,
            activ="leaky_relu",
            upsample=False,
            normalize=False,
        )
        self.conv_block2 = ConvBlock(
            64,
            128,
            kernel_size=4,
            stride=2,
            padding=1,
            activ="leaky_relu",
            upsample=False,
            normalize=False,
        )
        self.conv_block3 = ConvBlock(
            128,
            256,
            kernel_size=4,
            stride=2,
            padding=1,
            activ="leaky_relu",
            upsample=False,
            normalize=False,
        )
        self.conv_block4 = ConvBlock(
            256,
            512,
            kernel_size=4,
            stride=2,
            padding=1,
            activ="leaky_relu",
            upsample=False,
            normalize=False,
        )
        self.conv_block5 = ConvBlock(
            512,
            1024,
            kernel_size=4,
            stride=2,
            padding=1,
            activ="leaky_relu",
            upsample=False,
            normalize=False,
        )
        self.conv_block6 = ConvBlock(
            1024,
            2048,
            kernel_size=4,
            stride=2,
            padding=1,
            activ="leaky_relu",
            upsample=False,
            normalize=False,
        )
        self.src_out = ConvBlock(
            2048,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            activ="leaky_relu",
            upsample=False,
            normalize=False,
        )
        self.cls_out = ConvBlock(
            2048,
            dom_dim,
            kernel_size=2,
            stride=1,
            padding=0,
            activ="leaky_relu",
            upsample=False,
            normalize=False,
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x_src = self.src_out(x)
        x_cls = self.cls_out(x)
        return x_src.mean(dim=(1, 2, 3)), x_cls.squeeze() # We leverage PatchGANs."


if __name__ == "__main__":
    img_size = 128
    src_image = torch.randn(2, 3, img_size, img_size)
    dom_dim = 17
    trg_dom = torch.randn(2, dom_dim)
    G = Generator(dom_dim=dom_dim)
    out = G(src_image=src_image, trg_dom=trg_dom)
    
    x = torch.randn(2, 3, img_size, img_size)
    D = Discriminator(dom_dim=dom_dim)
    src_out, cls_out = D(x)
    print(src_out.shape, cls_out.shape)

    # "We use an $n$-dimensional one-hot vector to represent $m$, with $n$ being the number of datasets."
    batch_size = 4
    n_datasets = 2
    m = torch.randn((batch_size, n_datasets))

    # "We randomly generate the target domain label c so that G learns to flexibly translate the input image. We"