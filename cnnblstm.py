import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p):
        super(Conv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.end_bn = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
    def forward(self, h0):
        h1 = self.conv2d(h0)
        h1 = self.end_bn( h1 )
        h1 = self.leaky_relu(h1)
        return h1

class CNN_BLSTM(nn.Module):
    def __init__(self, in_dim, hid_dim, layer, do_ratio):
        super().__init__()
        self.in_dim = in_dim
        self.in_BN = nn.BatchNorm2d(1)
        self.conv1 = Conv(1, 30, (15,5), (1,1), (7,2))
        self.conv2 = Conv(30, 60, (15,5), (1,1), (7,2))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.cnn_IN = nn.InstanceNorm2d(1)
        self.depthwise = nn.Conv2d(60,1,1,1,0)
        self.linear_In = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(do_ratio)
        self.blstm = nn.LSTM(input_size=hid_dim, hidden_size=hid_dim, num_layers=layer, batch_first=True, dropout=do_ratio, bidirectional=True)
        self.linear_out = nn.Linear(hid_dim*2, in_dim*2)

    def forward(self, x):
        h = self.in_BN(x)
        h = self.conv1(h)
        h = self.conv2(h)
        cnn_out = self.leaky_relu(self.cnn_IN(self.depthwise(h)))
        h = self.leaky_relu(self.dropout(self.linear_In(cnn_out)))
        shape = h.shape
        h, (hn, cn) = self.blstm(h.reshape(shape[0],shape[2],shape[3]),None)
        shape = h.shape
        G2 = self.linear_out(h.reshape(shape[0],1,shape[1],shape[2]))
        Gr, Gi = torch.split(G2, self.in_dim, 3)
        return Gr, Gi