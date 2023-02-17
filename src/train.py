import torch
import torch.nn as nn

from segmentation_models_pytorch.losses import JaccardLoss

from model import UNet

import pandas as pd

import config
from dataset import get_loaders
from utils import save_model, FocalLoss, load_model
from engine import train, evaluate

def trainer(save_best=False, load=False):
    batch = pd.read_csv("../dataset/preprocessed_data.csv")

    train_loader, val_loader = get_loaders(batch)

    model = UNet(config.INP_CHANNELS, config.NUM_CLASSES)
    model.to(config.DEVICE)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr = config.LR)
    optimizer = torch.optim.Adam(model.parameters(), lr= config.LR)
    
    loss_fn = JaccardLoss(mode='multiclass')
    #loss_fn = FocalLoss(gamma=3)

    if load:
        model_state, optimizer_state, epoch_done = load_model()
        epoch_done+=1
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

        print(f"Continuing Training after {epoch_done} epochs")
    else:
        epoch_done=0

    best_loss = 10

    for epoch in range(epoch_done, config.EPOCH+epoch_done):
        print(f"Epoch {epoch}:")
        train_loss, train_score = train(train_loader, model, optimizer, loss_fn)
        val_loss, val_score = evaluate(val_loader, model, loss_fn)

        if save_best: #and val_loss<best_loss:
            best_loss = val_loss
            save_model(epoch, model, optimizer, val_loss)

        print(f"Training: Loss = {train_loss} Dice Score = {train_score},\
             Validataion: Loss = {val_loss} Dice Score = {val_score}")

if __name__ == "__main__":
    trainer(save_best=True, load=True)