import multichannel_resnet
from multichannel_resnet import get_arch as Resnet

resnet34_4_channel = Resnet(34, 4)

model = resnet34_4_channel(True)
print("New input channels : ", model.conv1.in_channels)