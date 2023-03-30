import numpy as np
from torch import nn
from tools.loadWeights import load_conv, load_conv_batch_norm


class GlobalAvgPool2d(nn.Module):
    """
    Class implements 2D global average pooling.
    It is necessary to train the classifier Darknet19 after convolutional layers.
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        height = x.data.size(2)
        width = x.data.size(3)
        layer = nn.AvgPool2d(kernel_size=height, stride=width)
        return layer(x)


class Darknet19(nn.Module):
    """
    Base class of YOLOv2 architecture.
    It contains 19 convolutional layers and is also used in YOLO
    with pretrained weights on classification.

    Note: Use load_weights() on this class for better training
    """
    def __init__(self):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.conv_block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1, padding=0, bias=True),

            # Global average pooling
            GlobalAvgPool2d(),

            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

    def load_weights(self, weightfile):
        """
        Loading weights to Darknet19.
        param: binary file of weights.

        Note: Model target weight size should be 20856776.
        weight file after converting to numpy should have size 20856776, therefore, it should be noted that the
        first 4 indexes are considered as the heading.
        """

        with open(weightfile, 'rb') as fp:
            header = np.fromfile(fp, count=4, dtype=np.int32)
            buf = np.fromfile(fp, dtype=np.float32)
            start = 0

            # load weights to convolution block 1
            for num_layer, layer in enumerate(self.conv_block1):
                if isinstance(layer, nn.modules.conv.Conv2d):
                    conv_layer = self.conv_block1[num_layer]
                    batch_norm_layer = self.conv_block1[num_layer + 1]
                    start = load_conv_batch_norm(buf, start, conv_layer, batch_norm_layer)

            # load weights to convolution block 2
            for num_layer, layer in enumerate(self.conv_block2):
                if isinstance(layer, nn.modules.conv.Conv2d):
                    conv_layer = self.conv_block2[num_layer]
                    batch_norm_layer = self.conv_block2[num_layer + 1]
                    start = load_conv_batch_norm(buf, start, conv_layer, batch_norm_layer)

            # load weights to output layer
            conv_layer = self.classifier[0]
            start = load_conv(buf, start, conv_layer)

            if start == buf.size:
                print("Darknet19 weight file upload successfully")
