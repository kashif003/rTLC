# this file will be used to train and validate the model.

import os
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from configs import configs
config = configs()
from utils import plot_random_batch
from tqdm import tqdm
#TODO need to add the self stopping
def train(model, dataloader, config):

    model.train()
    optimizer = AdamW(model.parameters(), lr= config["lr"])
    mse_Loss = MSELoss()
    total_loss = []
    for epoch in tqdm(range(config["epochs"])):
        model.train()
        for batch in dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mse_Loss(outputs, inputs)
            loss.backward()
            optimizer.step() 
        total_loss.append(loss)
        if epoch % 2==0:
            print("saving the figure:", epoch)
            plot_random_batch(model, dataloader)
