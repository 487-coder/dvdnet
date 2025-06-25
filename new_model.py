import torch
import torch.nn as nn

class Spatial_block(nn.Module):
    def __init__(self, input_channels = 3):   # RBG
        super(Spatial_block, self).__init__()
        self.downscale = nn.Unfold(kernel_size=(2, 2), stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=input_channels*4 + input_channels,
                                             out_channels= 96, kernel_size=3, stride=1,
                                             padding=1, bias=False),
                                   nn.ReLU(inplace= True))
        self.conv2 = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3,
                                             stride=1,padding=1, bias=False),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU(inplace=True)) for _ in range(10)
                                   ])
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels= input_channels *4,
                                             kernel_size=3, stride=1,padding=1, bias=False))
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.reset_parameters()
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
    def reset_parameters(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)
    def forward(self, x, noise_map):
        B, C, H, W = x.size()
        x1 = self.downscale(x)
        x2 = x1.reshape(B, C*4, H//2, W//2)
        x3 = torch.cat((noise_map[:,:,::2,::2], x2), dim=1)
        x4 = self.conv1(x3)
        x4 = self.conv2(x4)
        x4 = self.conv3(x4)
        x5= self.pixel_shuffle(x4)
        x = x - x5
        return x


class Temporal_block(nn.Module):
    def __init__(self,num_input_frames, frame_channels = 3):
        super(Temporal_block, self).__init__()
        self.downscale = nn.Sequential(nn.Conv2d(in_channels=int((num_input_frames+1)*frame_channels),
                                                 out_channels=96,kernel_size= 5, stride=2, padding=2,
                                                 bias=False),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(in_channels= 96,out_channels=96,kernel_size=1,
                               stride=1, padding=0,bias=False)
        self.conv2 = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3,
                                    stride=1, padding=1, bias=False),
                          nn.BatchNorm2d(96),
                          nn.ReLU(inplace=True)) for _ in range(4)
        ])
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=96,
                                             kernel_size=3, padding=1, stride=1,bias=False),
                                      nn.BatchNorm2d(96),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=96, out_channels= frame_channels*4,
                                                kernel_size=3, padding=1, stride=1, bias=False))
        self.pixelshuffle = nn.PixelShuffle(2)
        self.reset_parameters()

    @staticmethod
    def weight_init(m):
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_parameters(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)
    def forward(self, x, noise_map):
        B, C, H, W = x.size()
        x1 = torch.cat((noise_map, x),1)
        x1 = self.downscale(x1)
        x2 = self.conv1(x1)
        x1 = self.conv2(x1)
        x3 = self.conv3(x1+x2)
        x3 = self.pixelshuffle(x3)
        return x3
