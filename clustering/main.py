import pickle
from utils import Custom_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
# loading  the model.
with open("dataset_SR.pkl", "rb") as file:
    data= pickle.load(file)

print("[INFO] NUmber of samples:",len(data))
print("[INFO] Dataset type:", type(data))
# print(data[list(data.keys())[1]]["label"])


import random, torch
torch.manual_seed(32)
dataset = Custom_dataset(data)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, drop_last=False)

print("[INFO] number of batchs in dataloader:", len(dataloader))
