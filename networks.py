import torch
import torch.nn as nn
import random
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from config import get_config

# Functions
##############################################################################
def get_network(stage, config):
    if stage == "VGG":
        return VGG_net(config)

class VGG_net(nn.Module):
    def __init__(self, config, im_size=256):
        super(VGG_net, self).__init__()
        self.feature_extract, mid_channel = self.make_layers(config.cfg, im_size)
        self.classifier = self.make_classifier(mid_channel, config.fc)
        print(self.feature_extract)
        self._initialize_weights()

    def make_layers(self, cfg, im_size, batch_norm=True):
        layers = []
        in_channels = 3

        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                im_size = im_size // 2
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=2)
                im_size = im_size // 2
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers), im_size ** 2 * in_channels

    def make_classifier(self, mid_channel, fc):
        layers = []
        in_channels = mid_channel

        for v in fc:
            layers += [nn.Linear(in_channels, v)]
            if v != fc[-1]:
                layers += [nn.ReLU(inplace=True)]
                layers += [nn.Dropout()]
            in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extract(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == "__main__":
    config = get_network("VGG", get_config())
