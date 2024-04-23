import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aspp import ASPP

class Model(nn.Module):
    def __init__(self, resnet, num_classes):
        super(Model, self).__init__()


        # Encoder

        # Load pretrained ResNet layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        # ASPP module
        self.aspp = ASPP(in_channels=1024, out_channels=256)  # Assuming ResNet-50
        

        # Decoder

        # Low level features
        self.backbone2 = nn.Sequential(*list(resnet.children())[0:5])
        # Additional convolutional layer to adjust output channels
        self.conv = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # batch_norm
        self.bn1 = nn.BatchNorm2d(48)
        # Additional convolutional layer to adjust output channels
        self.conv2 = nn.Conv2d(48 + 256, 256, kernel_size=3, bias=False, padding='same')
        # batch_norm
        self.bn2 = nn.BatchNorm2d(256)
        # Output convolutional layer
        self.conv_output = nn.Conv2d(256, num_classes, kernel_size=1, bias=False, padding='same')

    
    def encoder(self, x):
        # Apply ASPP module
        resnet_output = self.backbone(x)
        aspp_output = self.aspp(resnet_output)

        return aspp_output
    

    def decoder(self, x, encoder_output):

        # Upsample to match the original input size
        upsample1 = self.upsample(encoder_output)

        features = self.backbone2(x)
        conv1 = self.conv(features)
        conv1 = F.relu(self.bn1(conv1))

        concat = torch.cat((conv1, upsample1), dim=1)

        conv2 = self.conv2(concat)
        conv2 = F.relu(self.bn2(conv2))
        upsample2 = self.upsample(conv2)

        decoder_output = self.conv_output(upsample2)
        decoder_output = F.sigmoid(decoder_output)

        return decoder_output

    def forward(self, x):
        # Extract features using the backbone
        encoder_output = self.encoder(x)

        decoder_output = self.decoder(x, encoder_output)
        
        return decoder_output
