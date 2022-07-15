"""
This python script converts the network into Script Module
"""
import torch
from torch import jit
from torchvision import models

# Download and load the pre-trained model
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

example_input = torch.rand(1, 3, 224, 224)
# fcn(example_input)
script_module = torch.jit.trace(fcn, example_input, strict=False)
# fcn.save("./resnet101.pt")
torch.save(fcn.state_dict(), "./resnet101_inference_model.pt")
