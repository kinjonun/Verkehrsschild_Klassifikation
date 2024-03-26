import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import cv2
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import os
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Load the data set
data_dir = '/home/zhensun/Downloads/gtrsb'
train_path = '/home/zhensun/Downloads/gtrsb/Train'
test_path = '/home/zhensun/Downloads/gtrsb/Test'
train_csv = pd.read_csv(data_dir + '/Train.csv')
test_csv = pd.read_csv(data_dir + '/Test.csv')

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

NUM_CATEGORIES = len(os.listdir(train_path))
print(NUM_CATEGORIES)

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        label = torch.tensor(label)
        # label = F.one_hot(label, num_classes=NUM_CATEGORIES).float()
        if self.transform:
            image = self.transform(image)
        return image, label

images = []
labels = []

for i in range(NUM_CATEGORIES):
    path = data_dir + '/Train/' + str(i)
    image_paths = os.listdir(path)

    for img_path in image_paths:
        full_path = os.path.join(path, img_path)  
        image_np = mpimg.imread(full_path)
        image_pil = Image.fromarray(np.uint8(image_np * 255))
        images.append(image_pil)
        labels.append(i)



def split_data_into_train_and_val(images, labels):

    data = list(zip(images, labels))    # Pair up the image paths and labels
    random.shuffle(data)                # Shuffle the data to ensure a random split
    train_size = int(0.7 * len(data))   # 70% for training

    train_data = data[:train_size]      # Split the data
    val_data = data[train_size:]
    train_images, train_labels = zip(*train_data)
    val_images, val_labels = zip(*val_data)

    return train_images, train_labels, val_images, val_labels


train_images, train_labels, val_images, val_labels = split_data_into_train_and_val(images, labels)

transform = transforms.Compose([
    transforms.Resize([224, 224]),   # Input size of EfficientNet-B0
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_images, train_labels, transform=transform)     # Load the dataset
val_dataset = CustomDataset(val_images, val_labels, transform=transform)

# Create a DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


model = EfficientNet.from_pretrained("efficientnet-b0")

# replace classifier
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, NUM_CATEGORIES)
model = model.to(device)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


best_val_accuracy = 0.0                         # initial best accuracy
best_model_path = 'best_model.pth'              # set model saving path

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    total_batches = len(train_dataloader)       # 858


    # tqdm progress bar for training
    train_progress_bar = tqdm(enumerate(train_dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{num_epochs}, Training")

    for i, (data, labels) in train_progress_bar:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_progress_bar.set_postfix(loss=(train_loss / (i + 1)))    # display the training loss in real time

    train_avg_loss = train_loss / total_batches
    print(f'Epoch {epoch+1}, Training Loss: {train_avg_loss}')

    # validation stage
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_batches = len(val_dataloader)

    # tqdm progress bar for validation
    val_progress_bar = tqdm(enumerate(val_dataloader), total=val_batches, desc=f"Epoch {epoch+1}/{num_epochs}, Validation")

    with torch.no_grad():
        for i, (data, labels) in val_progress_bar:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)               # Returns the maximum value predicted for each sample and its index
            val_total += labels.size(0)                             # size of a batch
            val_correct += (predicted == labels).sum().item()
            val_progress_bar.set_postfix(val_loss=(val_loss / (i + 1)))

    val_avg_loss = val_loss / val_batches
    val_accuracy = 100 * val_correct / val_total
    print(f'Epoch {epoch+1}, Validation Loss: {val_avg_loss}, Validation Accuracy: {val_accuracy:.2f}%')

    # save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'New best model saved at Epoch {epoch+1} with Validation Accuracy: {val_accuracy:.2f}%')
