# This implementation is based on the ResNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, models, transforms
import cv2
from torch2trt import torch2trt
from torch2trt import TRTModule


class ResNetModel:

    dictionary = {
        "Contempt": 7,
        "Anger": 6,
        "Disgust": 5,
        "Fear": 4,
        "Surprise": 3,
        "Sadness": 2,
        "Happiness": 1,
        "Neutral": 0,
    }

    def __init__(self, size=224, mode='gpu'):
        resnet50_model = ResNet('resnet18', num_cls=8)
        PATH = './models/resnet18.224.pth'
        if torch.cuda.is_available() and mode == 'gpu':
            checkpoint = torch.load(PATH)
        else:
            checkpoint = torch.load(PATH, map_location="cpu")

        resnet50_model.load_state_dict(checkpoint['model_state_dict'])
        resnet50_model.eval()
        if mode == 'gpu':
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.resnet50_model = resnet50_model.to(self.device)

        self.label_map = dict((v, k) for k, v in self.dictionary.items())
        self.size = size

        self.model_trt = TRTModule()
        PATH = './models/resnet18.224.trt.pth'
        self.model_trt.load_state_dict(torch.load(PATH))
        self.model_trt.eval().cuda()

    def image_loader(self, image):
        loader = transforms.Compose([transforms.ToTensor()])
        image = loader(image).float()
        image = image.unsqueeze(0)
        return image

    def resize_image(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image

    def face_expression(self, image):
        resized_image = self.resize_image(image)
        tensor_image = self.image_loader(resized_image)
        tensor_image = tensor_image.to(self.device)
        with torch.no_grad():
            outputs = self.resnet50_model.forward(tensor_image)
            outputs2 = self.model_trt(tensor_image)
            print("{}".format(outputs))
            print("{}".format(outputs2))
            _, predicted = torch.max(outputs, 1)
            idx = predicted.item()
            face_expression = self.label_map[idx]
            return face_expression


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

    def forward(self, input_image):
        cls = self.model(input_image)
        return cls
