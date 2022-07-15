"""
This python script converts the network into Script Module
"""
import torch
from torch import jit
from torchvision import models

# Download and load the pre-trained model
pretrained_net = models.vgg19(pretrained=True)

style_layers, content_layers = [0, 5, 10, 19, 28], [25]

# Save the model except the final two Layers
net = torch.nn.Sequential(*[pretrained_net.features[i] for i in
                            range(max(content_layers + style_layers) + 1)])

example_input = torch.rand(1, 3, 300, 450)
print(net(example_input).shape)
script_module = torch.jit.trace(net, example_input)
script_module.save('vgg19_neural_style.pt')
