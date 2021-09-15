import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.models.vgg import vgg16_bn

def transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def extractor():
    model = vgg16_bn(pretrained=True)
    return nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        model.classifier[0]
    )
