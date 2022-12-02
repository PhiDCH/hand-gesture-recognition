import torch
import torch.nn as nn
import torchvision


class UpsampleCBR(torch.nn.Sequential):
    def __init__(self, input_channels, output_channels, count=1, num_flat=0):
        layers = []
        for i in range(count):
            if i == 0:
                inch = input_channels
            else:
                inch = output_channels

            layers += [
                torch.nn.ConvTranspose2d(
                    inch, output_channels, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU()
            ]
            for i in range(num_flat):
                layers += [
                    torch.nn.Conv2d(output_channels, output_channels,
                                    kernel_size=3, stride=1, padding=1),
                    torch.nn.BatchNorm2d(output_channels),
                    torch.nn.ReLU()
                ]

        super(UpsampleCBR, self).__init__(*layers)


class CmapPafHeadAttention(torch.nn.Module):
    def __init__(self, input_channels, cmap_channels, paf_channels, upsample_channels=256, num_upsample=0, num_flat=0):
        super(CmapPafHeadAttention, self).__init__()
        self.cmap_up = UpsampleCBR(
            input_channels, upsample_channels, num_upsample, num_flat)
        self.paf_up = UpsampleCBR(
            input_channels, upsample_channels, num_upsample, num_flat)
        self.cmap_att = torch.nn.Conv2d(
            upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
        self.paf_att = torch.nn.Conv2d(
            upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)

        self.cmap_conv = torch.nn.Conv2d(
            upsample_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
        self.paf_conv = torch.nn.Conv2d(
            upsample_channels, paf_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        xc = self.cmap_up(x)
        ac = torch.sigmoid(self.cmap_att(xc))

        xp = self.paf_up(x)
        ap = torch.tanh(self.paf_att(xp))

        return self.cmap_conv(xc * ac), self.paf_conv(xp * ap)


class ResNetBackbone(torch.nn.Module):

    def __init__(self, resnet):
        super(ResNetBackbone, self).__init__()
        self.resnet = resnet

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)  # /4
        x = self.resnet.layer2(x)  # /8
        x = self.resnet.layer3(x)  # /16
        x = self.resnet.layer4(x)  # /32

        return x


def _resnet_pose_att(cmap_channels, paf_channels, upsample_channels, resnet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        ResNetBackbone(resnet),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels,
                             upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model


def resnet18_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    # resnet = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
    return _resnet_pose_att(cmap_channels, paf_channels, upsample_channels, resnet, 512, num_upsample, num_flat)


if __name__ == '__main__':
    import json

    with open('preprocess/hand_pose.json', 'r') as f:
        hand_pose = json.load(f)
    num_parts = len(hand_pose['keypoints'])
    num_links = len(hand_pose['skeleton'])
    model = resnet18_baseline_att(num_parts, 2 * num_links).eval()

    MODEL_WEIGHTS = 'model/hand_pose_resnet18_att_244_244.pth'
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
