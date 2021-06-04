import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=False, instance_norm=False):
    """
    Creates a convolutional layer, with optional batch / instance normalization.
    """

    # Add layers
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=False)
    layers.append(conv_layer)

    # Batch normalization
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    # Instance normalization
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()
        """
        Input is RGB image (256x256x3) while output is a single value

        determine size = [(W−K+2P)/S]+1
        W: input=256
        K: kernel_size=4
        P: padding=1
        S: stride=2
        """

        # convolutional layers, increasing in depth
        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4)  # (128, 128, 64)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4,
                          instance_norm=True)  # (64, 64, 128)
        self.conv3 = conv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4,
                          instance_norm=True)  # (32, 32, 256)
        self.conv4 = conv(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=4,
                          instance_norm=True)  # (16, 16, 512)
        self.conv5 = conv(in_channels=conv_dim * 8, out_channels=conv_dim * 8, kernel_size=4,
                          batch_norm=True)  # (8, 8, 512)

        # final classification layer
        self.conv6 = conv(conv_dim * 8, out_channels=1, kernel_size=4, stride=1)  # (8, 8, 1)

    def forward(self, x):
        # leaky relu applied to all conv layers but last
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2)
        #       out = F.leaky_relu(self.conv5(out), negative_slope=0.2)

        # classification layer (--> depending on the loss function we might want to use an activation function here, e.g. sigmoid)
        out = self.conv6(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        """
        Residual blocks help the model to effectively learn the transformation from one domain to another. 
        """
        self.conv1 = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1,
                          instance_norm=True)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1,
                          instance_norm=True)

    def forward(self, x):
        out_1 = F.relu(self.conv1(x))
        out_2 = x + self.conv2(out_1)
        return out_2


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=False, instance_norm=False,
           dropout=False, dropout_ratio=0.5):
    """
    Creates a transpose convolutional layer, with optional batch / instance normalization. Select either batch OR instance normalization.
    """

    # Add layers
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

    # Batch normalization
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    # Instance normalization
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))

    # Dropout
    if dropout:
        layers.append(nn.Dropout2d(dropout_ratio))

    return nn.Sequential(*layers)


class CycleGenerator(nn.Module):

    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()
        """
        Input is RGB image (256x256x3) while output is a single value

        determine size = [(W−K+2P)/S]+1
        W: input=256
        K: kernel_size=4
        P: padding=1
        S: stride=2
        """

        # Encoder layers
        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4)  # (128, 128, 64)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4,
                          instance_norm=True)  # (64, 64, 128)
        self.conv3 = conv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4,
                          instance_norm=True)  # (32, 32, 256)

        # Residual blocks (number depends on input parameter)
        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim * 4))
        self.res_blocks = nn.Sequential(*res_layers)

        # Decoder layers
        self.deconv4 = deconv(in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=4,
                              instance_norm=True)  # (64, 64, 128)
        self.deconv5 = deconv(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4,
                              instance_norm=True)  # (128, 128, 64)
        self.deconv6 = deconv(in_channels=conv_dim, out_channels=3, kernel_size=4, instance_norm=True)  # (256, 256, 3)

    def forward(self, x):
        """
        Given an image x, returns a transformed image.
        """

        # Encoder
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2)  # (128, 128, 64)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)  # (64, 64, 128)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)  # (32, 32, 256)

        # Residual blocks
        out = self.res_blocks(out)

        # Decoder
        out = F.leaky_relu(self.deconv4(out), negative_slope=0.2)  # (64, 64, 128)
        out = F.leaky_relu(self.deconv5(out), negative_slope=0.2)  # (128, 128, 64)
        out = torch.tanh(self.deconv6(out))  # (256, 256, 3)

        return out

def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model.
    The weights are taken from a normal distribution with mean = 0, std dev = 0.02.
    Param m: A module or layer in a network
    """
    # classname will be something like: `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__

    # normal distribution with given paramters
    std_dev = 0.02
    mean = 0.0

    # Initialize conv layer
    if hasattr(m, 'weight') and (classname.find('Conv') != -1):
        init.normal_(m.weight.data, mean, std_dev)


def build_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """
    Builds generators G_XtoY & G_YtoX and discriminators D_X & D_Y
    """

    # Generators
    G_XtoY = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    G_YtoX = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)

    # Discriminators
    D_X = Discriminator(conv_dim=d_conv_dim)  # Y-->X
    D_Y = Discriminator(conv_dim=d_conv_dim)  # X-->Y

    # Weight initialization
    G_XtoY.apply(weights_init_normal)
    G_YtoX.apply(weights_init_normal)
    D_X.apply(weights_init_normal)
    D_Y.apply(weights_init_normal)

    # Moves models to GPU, if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y