import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        # ASPP modules
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=rates[0], dilation=rates[0])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=rates[1], dilation=rates[1])
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=rates[2], dilation=rates[2])
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.out_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)

        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.bn6 = nn.BatchNorm2d(out_channels)

        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # x = torch.Size([32, 1024, HEIGHT/16, WIDTH/16])
        # Apply ASPP operations
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(x)))
        out3 = F.relu(self.bn3(self.conv3(x)))
        out4 = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling        
        global_pool = F.adaptive_avg_pool2d(x, (1, 1)) # torch.Size([32, 1024, 1, 1])
        global_pool = self.conv5(global_pool) # torch.Size([32, 256, 1, 1])
        global_pool = self.bn5(global_pool)
        global_pool = F.relu(global_pool)        
        global_pool = F.interpolate(global_pool, size=out1.shape[2:], mode='bilinear', align_corners=True) # torch.Size([32, 256, 2, 2])
        
        # Concatenate results
        out = torch.cat((out1, out2, out3, out4, global_pool), dim=1)
        out = self.out_conv(out)
        out = F.relu(self.bn6(out))
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)