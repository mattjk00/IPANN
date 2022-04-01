import platform
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset



class Net(nn.Module):
  def __init__(self, INPUT_SIZE=28800, OUTPUT_SIZE=63):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 3, bias=False)
    self.conv2 = nn.Conv2d(6, 32, 3)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(INPUT_SIZE, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, OUTPUT_SIZE)
  
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
    
SPLIT = '/' if platform.system() != 'Windows' else '\\'

class SymbolDataset(Dataset):
    def __init__(self, image_paths, class_to_index, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_index = class_to_index
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split(SPLIT)[-2]
        label = self.class_to_index[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label

class PredictSymbolDataset(Dataset):
    def __init__(self, image_paths,transform=False):
        self.image_paths = image_paths
        self.transform = transform
        #self.class_to_index = class_to_index
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] < 128 or image.shape[1] < 128:
            image = cv2.resize(image, (128, 128))
        
        #label = image_filepath.split(SPLIT)[-2]
        #label = 0#self.class_to_index[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        #label = 0
        
        return image#, label