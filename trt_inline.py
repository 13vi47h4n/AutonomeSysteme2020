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


class TRTModel:

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

    def __init__(self, size=224):
        self.model = ResNet('resnet50')
        PATH = './models/resnet50.224.pth'
        self.model.load_state_dict(torch.load(PATH)['model_state_dict'])
        self.model.eval().cuda()

        #create dummy data
        data = torch.ones((1, 3, 224, 224)).cuda()

        # convert to TensorRT feeding sample data as input
        self.model_trt = torch2trt(self.model, [data])

        print("{}".format(self.model.forward(data)))
        print("{}".format(self.model_trt(data)))

        self.label_map = dict((v, k) for k, v in self.dictionary.items())
        self.size = size

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
        tensor_image = tensor_image.cuda()
        with torch.no_grad():
            outputs = self.model_trt(tensor_image)
            _, predicted = torch.max(outputs, 1)
            print("{} \n{}".format(outputs,self.model.forward(tensor_image)))
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
