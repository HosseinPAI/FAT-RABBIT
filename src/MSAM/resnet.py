import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Resnet20(nn.Module):

    def __init__(self, num_classes=10):
        super(Resnet20, self).__init__()
        self.num_classes = num_classes

        # input blocks
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.shortcut = nn.Sequential()

        # Block 1
        self.Block1_Layer1 = self._make_layer(in_ch=16, out_ch=16, kernel=3, stride1=1, stride2=1, padding=1)
        self.Block1_Layer2 = self._make_layer(in_ch=16, out_ch=16, kernel=3, stride1=1, stride2=1, padding=1)
        self.Block1_Layer3 = self._make_layer(in_ch=16, out_ch=16, kernel=3, stride1=1, stride2=1, padding=1)

        # Block 2
        self.Block2_Layer1 = self._make_layer(in_ch=16, out_ch=32, kernel=3, stride1=2, stride2=1, padding=1)
        self.padding1 = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 32 // 4, 32 // 4),
                                                     "constant", 0))
        self.Block2_Layer2 = self._make_layer(in_ch=32, out_ch=32, kernel=3, stride1=1, stride2=1, padding=1)
        self.Block2_Layer3 = self._make_layer(in_ch=32, out_ch=32, kernel=3, stride1=1, stride2=1, padding=1)

        # Block 3
        self.Block3_Layer1 = self._make_layer(in_ch=32, out_ch=64, kernel=3, stride1=2, stride2=1, padding=1)
        self.padding2 = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 64 // 4, 64 // 4),
                                                     "constant", 0))
        self.Block3_Layer2 = self._make_layer(in_ch=64, out_ch=64, kernel=3, stride1=1, stride2=1, padding=1)
        self.Block3_Layer3 = self._make_layer(in_ch=64, out_ch=64, kernel=3, stride1=1, stride2=1, padding=1)

        # Linear
        self.linear = nn.Linear(in_features=64, out_features=self.num_classes)

        # Relu Layer
        self.relu = nn.ReLU()

    def _make_layer(self, in_ch=16, out_ch=16, kernel=3, stride1=1, stride2=1, padding=1):

        self.Layer = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride1,
                      padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel, stride=stride2,
                      padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_ch)
        )

        return self.Layer

    def forward(self, x):
        # First Layer
        out = self.relu(self.bn1(self.conv1(x)))
        # Block1
        out = self.relu(self.Block1_Layer1(out) + self.shortcut(out))
        out = self.relu(self.Block1_Layer2(out) + self.shortcut(out))
        out = self.relu(self.Block1_Layer3(out) + self.shortcut(out))
        # Block2
        out = self.relu(self.Block2_Layer1(out) + self.padding1(out))
        out = self.relu(self.Block2_Layer2(out) + self.shortcut(out))
        out = self.relu(self.Block2_Layer3(out) + self.shortcut(out))
        # Block3
        out = self.relu(self.Block3_Layer1(out) + self.padding2(out))
        out = self.relu(self.Block3_Layer2(out) + self.shortcut(out))
        out = self.relu(self.Block3_Layer3(out) + self.shortcut(out))
        # FC
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = F.softmax(out, dim=1)
        #print(out.shape)

        return out
