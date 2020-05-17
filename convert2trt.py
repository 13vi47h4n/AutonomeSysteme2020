import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

resnet50_model = ResNet('resnet50')
PATH = './best.pth'
checkpoint = torch.load(PATH)

resnet50_model.load_state_dict(checkpoint['model_state_dict'])
resnet50_model.eval()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(resnet50_model, [x])

torch.save(model_trt.state_dict(), 'best_trt.pth')