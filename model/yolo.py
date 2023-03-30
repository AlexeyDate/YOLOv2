import torch
from torch import nn
from model.darknet import Darknet19


class Reorg(nn.Module):
    """
    Class implements reorganization layers in the YOLOv2.
    """
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        s = self.stride
        assert (x.data.dim() == 4)

        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)

        assert (H % s == 0)
        assert (W % s == 0)
        h, w = H // s, W // s

        x = x.view(B, C, h, s, w, s).transpose(3, 4).contiguous()
        x = x.view(B, C, h * w, s * s).transpose(2, 3).contiguous()
        x = x.view(B, C, s * s, h, w).transpose(1, 2).contiguous()
        x = x.view(B, s * s * C, h, w)
        return x


class YOLOv2(nn.Module):
    """
    Class implements the original architecture of the YOLOv2 model.
    This class contains Darknet19 model - it is CNN backbone of the YOLOv2 model
    """

    def __init__(self, num_anchors=5, num_classes=1000, device='cpu', darknet_weights=None):
        """
        param: num_anchors - number of anchor boxes
        param: num_classes - number of classes (default = 1000)
        param: device - device of model (default = 'cpu', available = 'gpu')
        param: darknet_weights - weight file of Darknet19 backbon model (default = None)
        """

        super(YOLOv2, self).__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.device = device

        self.darknet = Darknet19().to(self.device)

        if darknet_weights is not None:
            self.darknet.load_weights(darknet_weights)

        # delete layer from Darknet classifier (easier to perceive)
        self.darknet.classifier = nn.Sequential()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.route = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.detection = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=self.num_anchors * (5 + num_classes), kernel_size=1, stride=1,
                      padding=0, bias=False)
        )

    def forward(self, x):
        x1 = self.darknet.conv_block1(x)
        x2 = self.darknet.conv_block2(x1)
        x2 = self.conv_block(x2)
        x1 = self.route(x1)
        reorg = Reorg(stride=2)
        x1 = reorg(x1)
        x = torch.cat((x2, x1), dim=1)
        x = self.detection(x)
        return x.permute(0, 2, 3, 1)
