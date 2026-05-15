import pickle
from utils import Custom_dataset
from utils import plot_random_batch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
# loading  the model.
with open("data/dataset_SR.pkl", "rb") as file:
    data= pickle.load(file)

#print("[INFO] NUmber of samples:",len(data))
#print("[INFO] Dataset type:", type(data))
# print(data[list(data.keys())[1]]["label"])
import random, torch
torch.manual_seed(32)
dataset = Custom_dataset(data)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, drop_last=False)

for sample in dataloader:
    trail_input = sample[0]
    break

# ---
from model import Autoencoder
trail_encoder = Autoencoder()
trail_encoder.eval()

with torch.no_grad():
    output = trail_encoder(trail_input)
print("\n Input:\n",trail_input.shape)
print("\n output:\n",output[0].shape)
 
#---
from torch.nn import MSELoss
re_loss = MSELoss()
loss = re_loss(trail_input, output[0])
print("\n Loss:", loss.item())

#--

from train_valid import train
from configs import configs

config = configs()
train(trail_encoder,dataloader,config)
