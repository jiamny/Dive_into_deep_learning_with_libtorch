"""
This python script converts the network into Script Module
"""
import torch
from torch import jit
from torchvision import models
from torchvision.models import vgg19, VGG19_Weights

# Download and load the pre-trained model
pretrained_net = models.vgg19(weights=VGG19_Weights.DEFAULT).features

for param in pretrained_net.parameters():
        param.requires_grad = False

#style_layers, content_layers = [0, 5, 10, 19, 28], [25]

# Save the model
#net = torch.nn.Sequential(*[pretrained_net.features[i] for i in
#                            range(max(content_layers + style_layers) + 1)])
print(len(pretrained_net))
example_input = torch.rand(1, 3, 300, 450)
print(pretrained_net(example_input).shape)
script_module = torch.jit.trace(pretrained_net, example_input)
script_module.save('vgg19_neural_style.pt')

