import torchvision
import torch.nn as nn
from functools import partial

resnet_models = {18: torchvision.models.resnet18,
                 34: torchvision.models.resnet34,
                 50: torchvision.models.resnet18,
                 101: torchvision.models.resnet101,
                 152: torchvision.models.resnet152}

class Resnet_multichannel(nn.Module):
    def __init__(self, pretrained=True, encoder_depth=34, num_in_channels=4):
        super().__init__()
        
        if encoder_depth not in [18, 34, 50, 101, 152]:
            raise ValueError(f"Encoder depth {encoder_depth} specified does not match any existing Resnet models")
            
        model = resnet_models[encoder_depth](pretrained)
        
        ##For reference: layers to use (in order):
        # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        
        # This is the most important line of code here. This increases the number of in channels for our network
        self.conv1 = self.increase_channels(model.conv1, num_in_channels)
        
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc()
        
        return x
        
    def increase_channels(self, m, num_channels=None, copy_weights=0):


        """
        takes as input a Conv2d layer and returns the a Conv2d layer with `num_channels` input channels
        and all the previous weights copied into the new layer.
        """
        # number of input channels the new module should have
        new_in_channels = num_channels if num_channels is not None else m.in_channels + 1
        
        bias = False if m.bias is None else True
        
        # Creating new Conv2d layer
        new_m = nn.Conv2d(in_channels=new_in_channels, 
                          out_channels=m.out_channels, 
                          kernel_size=m.kernel_size, 
                          stride=m.stride, 
                          padding=m.padding,
                          bias=bias)
        
        # Copying the weights from the old to the new layer
        new_m.weight[:, :m.in_channels, :, :] = m.weight.clone()
        
        #Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
        for i in range(new_in_channels - m.in_channels):
            channel = m.in_channels + i
            new_m.weight[:, channel:channel+1, :, :] = m.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_m.weight = nn.Parameter(new_m.weight)

        return new_m
    
def get_arch(encoder_depth, num_in_channels):
    """
    Returns just an architecture which can then be called in the usual way.
    For example:
    resnet34_4_channel = get_arch(34, 4)
    model = resnet34_4_channel(True)
    """
    return partial(Resnet_multichannel, encoder_depth=encoder_depth, num_in_channels=num_in_channels)