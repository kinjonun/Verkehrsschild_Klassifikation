import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
from torchvision import datasets, transforms
from torchvision import models
import random
import os
from efficientnet_pytorch import EfficientNet

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

# Load the data set
data_dir = '/home/sun/archive'
train_path = '/home/sun/archive/Train'
test_path = '/home/sun/archive/Test'
train_csv = pd.read_csv(data_dir + '/Train.csv')
test_csv = pd.read_csv(data_dir + '/Test.csv')

transform = transforms.Compose([
    transforms.Resize([224, 224]),   # Input size of EfficientNet-B0
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = EfficientNet.from_name('efficientnet-b0', num_classes=43)
model.load_state_dict(torch.load("best_model.pth"))
# model = torch.load("best_model.pth")

model.to(device)

def predict_image(image, model):
    # transform image
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        # data, labels = data.to(device), labels.to(device)
        model.eval()
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()


test_image_paths = test_csv['Path'].sample(1).values
test_image = Image.open(os.path.join(data_dir, test_image_paths[0]))
predicted_class = predict_image(test_image, model)
print(f'Predicted class: {classes[predicted_class]}')
test_image.show()