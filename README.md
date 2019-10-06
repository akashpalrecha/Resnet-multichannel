# Resnet-multichannel
Contains a class to help use **Multichannel Pretrained Resnet Models** in PyTorch that take an *arbitary number of channels(> 3)* as input.
<br>
*The models implemented here carry much of transfer learning's benefits even when expanded to an arbitrary number of input channels. This is implemented by sharing the weights of existing trained 3 input kernels into the new input kernels that we add to the network.*
<br>
The whole process is simplified in a few lines of code : <br>
```python
import multichannel_resnet
from multichannel_resnet import get_arch as Resnet

#returns a callable that you can pass to libraries like fastai.
#Usage: Resnet(encoder_depth, number_of_desired_input_channels)
resnet34_4_channel = Resnet(34, 4)

# use resnet34_4_channels(False) to get a non pretrained model
model = resnet34_4_channel(True) 

print("New input channels : ", model.conv1.in_channels)
```
