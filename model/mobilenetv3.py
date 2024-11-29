import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MobileNetV3Encoder(nn.Module):
    """
    MobileNetV3Encoder based on torchvision's MobileNetV3-Small. Modified to
    use dilation and remove classifier, similar to the V2 version.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        state_dict = torch.load("./checkpoints/mobilenet_v3_small-047dcff4.pth",
                                map_location=device,
                                weights_only=True)
        base.load_state_dict(state_dict)

        self.features = base.features

        if in_channels != 3:
            self.features[0][0] = nn.Conv2d(
                in_channels, 16, 3, 2, 1, bias=False)

        self.features[-3].block[0].stride = (1, 1)
        for layer in self.features[-2:]:
            if hasattr(layer, 'block'):
                layer.block[0].dilation = (2, 2)
                layer.block[0].padding = (2, 2)

    def forward(self, x):
        features = []
        x0 = x
        features.append(x0)

        x = self.features[0](x)
        x = self.features[1](x)
        x1 = x
        features.append(x1)

        x = self.features[2](x)
        x = self.features[3](x)
        x2 = x
        features.append(x2)

        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x3 = x
        features.append(x3)

        for i in range(7, len(self.features)):
            x = self.features[i](x)
        x4 = x
        features.append(x4)

        return x4, x3, x2, x1, x0


if __name__ == "__main__":
    model = MobileNetV3Encoder(in_channels=3)

    x = torch.randn(1, 3, 224, 224)
    outputs = model(x)

    for i, feat in enumerate(outputs):
        print(f"Feature {i} shape:", feat.shape)
