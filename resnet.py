# This implementation is based on the ResNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from __future__ import print_function, division
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T

__all__ = ['ResNet']

dictionary = {
     "Contempt":7,
     "Anger":6,
     "Disgust":5,
     "Fear":4,
     "Surprise":3,
     "Sadness":2,
     "Happiness":1,
     "Neutral": 0,
}

label_map = dict((v,k) for k, v in dictionary.items())

def image_loader(image):
    """load image, returns cuda tensor"""
    #image = Image.open(image)
    #image = cv2.resize(image, (224, 224)) 
    loader = transforms.Compose([ transforms.ToTensor()])
    image = loader(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU

def crop_image(image):
    image = cv2.resize(image, (224, 224)) 
    return image

def face_expression(image_path):  
    image = cv2.imread(path)
    resnet50_model = ResNet('resnet50')
    PATH = './best.pth'
    checkpoint = torch.load(PATH)
    resnet50_model.load_state_dict(checkpoint['model_state_dict'])
    resnet50_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #resnet50_model = resnet50_model.to(device)
    image = crop_image(image)
    image = image_loader(image)
    input_image = {"image": image}
    with torch.no_grad():
        face_expressions = []
        outputs = resnet50_model.forward(input_image)
        _, predicted = torch.max(outputs, 1)
        idx = predicted.item()
        face_expressions.append(label_map[idx])
        print(face_expressions)
        return face_expressions

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
