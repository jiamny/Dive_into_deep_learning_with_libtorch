"""
This python script converts the network into Script Module
"""
import torch
import torchvision.models
from torch import jit
from torchvision import models

# Download and load the pre-trained model
model = models.resnet18(weights=torchvision.models.ResNet18_Weights)

# Set upgrading the gradients to False
# for param in model.parameters():
#	param.requires_grad = False

example_input = torch.rand(1, 3, 224, 224)
print(model(example_input).shape)

script_module = torch.jit.trace(model, example_input)
script_module.save('resnet18_jit_model.pt')

# Save the model except the final two Layers
resnet18 = torch.nn.Sequential(*list(model.children())[:-2])

example_input = torch.rand(1, 3, 320, 480)
print(resnet18(example_input).shape)

script_module = torch.jit.trace(resnet18, example_input)
script_module.save('resnet18_without_last_2_layers.pt')
