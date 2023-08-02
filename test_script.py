# dataset, dataloader
# train, evaluate
# test

from torch.utils.data import Dataset, DataLoader, random_split
from typing import Any
from torch import nn
import numpy as np
import torch
from torch import optim
from torchviz import make_dot

torch.manual_seed(0)
import random; random.seed(0)
np.random.seed(0)

class MyDataset(Dataset):
    def __init__(self):
        self.x = np.random.rand(100, 4, 6)
        self.y = self.x.sum(axis=(1,2))
        
    def __getitem__(self, index) -> Any:
        return torch.tensor(self.x[index],dtype=torch.float32), \
                        torch.tensor(self.y[index], dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
        

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Conv1d(in_channels=4, kernel_size=3,
                                out_channels=5, stride=2) 
        # 16,5,2
        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # self.layer3 = nn.Sequential(nn.Conv1d(in_channels=4, kernel_size=3,
        #                         out_channels=5, stride=2),
        #                             nn.Flatten(),
        #                             nn.ReLU(),
        #                             nn.Linear(10, 5),
        #                             nn.ReLU(),
        #                             nn.Linear(5, 1))
                                
                        
        self.loss = nn.MSELoss(reduction='mean')
        self.model_optimizer = optim.Adam(self.parameters(),
                                        lr=0.005)

    
    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = x.flatten(start_dim=1)        
        x = self.layer2(x)
        # x = self.layer3(x)
        return x
    
    def train_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.flatten(), y)  
        make_dot(y_hat, params=dict(self.named_parameters())).render("attached", format="png")
        return loss
    
    def step_optimizer(self, batch):
        self.train() 
        loss = self.train_step(batch)
        loss.backward() # computes gradient
        self.model_optimizer.step() # updates the model parameters with gradients calculated above
        self.model_optimizer.zero_grad() 
        return loss.item()
        
    @torch.no_grad()
    def val_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.flatten(), y)
        return loss.item()
    
def main():
    dataset = MyDataset()
    train_dataset, val_dataset = random_split(dataset, [80, 20])
    train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=80,
                                shuffle=False) 
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=20)
    model = MyNetwork()
    # model = model.to("cpu")
    
    max_epochs = 3
    val_freq = 1
    for i in range(max_epochs):
        
        for batch_idx, batch in enumerate(train_loader):
            train_loss = model.step_optimizer(batch) 
            print(f"batch idx: {batch_idx}, loss: {train_loss:0.2f}")
            
        if i%val_freq==0:
            model.eval()
            for eval_idx, batch in enumerate(val_loader):
                eval_loss = model.val_step(batch)
                print(f"eval idx: {eval_idx}, loss: {eval_loss:0.2f}")

        print(f"epoch {i} completed")
        
        
        
if __name__=="__main__":
    main()
        