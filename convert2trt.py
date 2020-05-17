import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import cv2

resnet50_model = ResNet('resnet50')
PATH = './best.pth'
checkpoint = torch.load(PATH)

resnet50_model.load_state_dict(checkpoint['model_state_dict'])
resnet50_model.eval()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(resnet50_model, [x])

torch.save(model_trt.state_dict(), 'best_trt.pth')


class ResNet(nn.Module):

    def __init__(
        self, arch, pretrained=False, progress=False, num_cls=8, **kwargs
    ):
        super().__init__()

        if(arch == 'resnet18'):
            self.model = torchvision.models.resnet18(num_classes=num_cls)
        elif(arch == 'resnet50'):
            self.model = torchvision.models.resnet50(num_classes=num_cls)
        else:
            raise NotImplementedError("Resnet to be implemented:", arch)

    def forward(self, input_dict):
        cls = self.model(input_dict["image"])
        return cls