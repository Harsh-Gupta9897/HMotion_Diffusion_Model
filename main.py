from typing import Optional, Tuple, Union, List
import math
import torch
from torch import nn
import torch
import torch.nn.functional as F

import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import transforms
from torchvision.datasets import MNIST, CIFAR10

import matplotlib.pyplot as plt
from tqdm import  trange
# import random
import pandas as pd
import numpy as np
import unet
from utils import motion_Nlength,sample,kps_generate
import renderer
from diffusion import DenoiseDiffusion

data = pd.read_json('data/normalized_uptown_funk.json')
cols = data.columns
data = np.array(data)
np.random.shuffle(data)

####### Hyperparameters ########
lr = 1e-5

batch_size= 16
N = 48
n_steps = 1000
num_epochs=400
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_set = motion_Nlength(data,N)
print("No of Motions: ",len(train_set),"\tMotion_Shape:", train_set[0].shape,"\tLength of Motion:",N)
train_loader = DataLoader(dataset=train_set,batch_size=batch_size, shuffle=True, pin_memory=True)

x = next(iter(train_loader))
print("\nbatch Sample:",x.shape)

   
def train(train_loader,optimizer,diffusion_model,device,num_epochs=5):
    losses = []
    for epoch in trange(num_epochs):
        for data in train_loader:
            
            # Move data to device
            data = data.to(device)

            optimizer.zero_grad()

            # Calculate loss
            loss = diffusion_model.loss(data)

            # Compute gradients
            loss.backward()

            # Take an optimization step
            optimizer.step()

            # Track the loss
            losses.append(loss.cpu().item())
            
        if((epoch+1)%20==0):
            print("Loss: ",losses[-1],'epoch:', epoch+1)
    fig = plt.figure(figsize=(15,10))
    plt.plot(losses)
    plt.savefig("losses")
    
    return losses




if __name__=='__main__':

    train_func=True
    if train_func:
        ######### Training #######
        eps_model = unet.MDM_UNetModel(image_size=16,in_channels=40
            ,model_channels= 32,
            out_channels=40,
            num_res_blocks= 4,
            attention_resolutions= 2).to(device)
        diffusion_model = DenoiseDiffusion(eps_model=eps_model,
                                        n_steps=n_steps,
                                        device=device)

        optimizer = torch.optim.AdamW(eps_model.parameters(), lr=0.0002)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1, last_epoch=- 1, verbose=False)
        losses = train(train_loader,optimizer,diffusion_model,device,num_epochs=num_epochs)
        torch.save(eps_model.state_dict(),"data/model_mdm2.pth")
    else:
        eps_model = unet.MDM_UNetModel(image_size=16,in_channels=40
            ,model_channels= 32,
            out_channels=40,
            num_res_blocks= 4,
            attention_resolutions= 2).to(device)
        eps_model.load_state_dict(torch.load("data/model_mdm.pth"))
        diffusion_model = DenoiseDiffusion(eps_model=eps_model,
                                        n_steps=n_steps,
                                        device=device)
            

    x = sample(diffusion_model,n_samples=64,device=device,n_steps=n_steps,N=N)
    print(x.shape)

    for i in range(10):
        y = x[i,:,:,:]
        print(y.shape,torch.max(y),torch.min(y))
        
        y[:,0,:] = ((y[:,0,:]-torch.min(y[:,0,:]))/(torch.max(y[:,0,:]-torch.min(y[:,0,:]))))*(427.980224609375-85.49298095703125)  + 85.49298095703125
        y[:,1,:] = ((y[:,1,:]-torch.min(y[:,1,:]))/(torch.max(y[:,1,:]-torch.min(y[:,1,:]))))*(538.4390258789062-39.429275512695312) + 39.429275512695312
        print(y.shape,torch.max(y),torch.min(y))
        result = kps_generate(cols,y)
        print(len(result[cols[0]]),len(result[cols[1]]),len(result[cols[0]][1]))
        renderer.render_seq(result,f"./data/demo_d_{i}.gif")

    x = next(iter(train_loader))
    print(torch.max(x),torch.min(x))

