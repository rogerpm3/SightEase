import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import transforms, models
from collections import OrderedDict
from PIL import Image


im_size = 224
transformation = transforms.Compose(
            [   transforms.Resize(256),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

def load_checkpoint(filepath, class_mapping):

    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        num_classes = len(class_mapping)

        if "resnet50" in checkpoint["arch"]:
            pattern_model = models.resnet50(pretrained=True)
            num_ftrs = pattern_model.fc.in_features
            classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc", nn.Linear(num_ftrs, num_classes)),
                    ("output", nn.LogSoftmax(dim=1)),
                ] ))
            pattern_model.fc = classifier

        for param in pattern_model.parameters():
            param.requires_grad = False

        pattern_model.class_to_idx = checkpoint["class_to_idx"]        
        pattern_model.load_state_dict(checkpoint["model_state_dict"])

        return pattern_model


class_mapping={'checked': 0, 'graphic': 1, 'plain': 2, 'stripe': 3}
get_label={0: 'checked', 1: 'graphic', 2: 'plain', 3: 'stripe'}
pattern_model = load_checkpoint('20240514_resnet50.pth', class_mapping)

pattern_model.eval()
img = Image.open('1931.jpg').convert("RGB")
img = transformation(img)
image = img.view([1, img.shape[0], img.shape[1], img.shape[2]])
with torch.no_grad():
    output = pattern_model.forward(image)
    probabilities = torch.exp(output)
    print(probabilities)
    predictions = probabilities.max(dim=1)[1]
print(predictions)
print(get_label[predictions.item()])