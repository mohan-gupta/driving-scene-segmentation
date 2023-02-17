import torch
import torch.nn as nn

from tqdm import tqdm

import config
from utils import smp_dice_score


def train(data_loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(data_loader)
    total_loss = 0
    dice_scores = 0
    for batch_idx, data in enumerate(loop):
        optimizer.zero_grad()
        score = 0

        images = data['images'].to(config.DEVICE)
        true_masks = data['masks'].to(config.DEVICE)

        masks = model(images)
        loss = loss_fn(masks, true_masks)
            
        with torch.no_grad():
            score = smp_dice_score(masks, true_masks)
            score = score.detach().cpu().item()
            dice_scores += score

        loss.backward()
        optimizer.step()

        batch_loss = loss.detach().cpu().item()
        total_loss += batch_loss

        loop.set_postfix(dict(
            loss = batch_loss,
            dice_score = score
        ))

    return round(total_loss/len(data_loader), 4), round(dice_scores/len(data_loader), 4)


def evaluate(data_loader, model, loss_fn):
    model.eval()
    loop = tqdm(data_loader)
    dice_scores = 0
    with torch.no_grad():
        total_loss = 0
        for batch_idx, data in enumerate(loop):

            images = data['images'].to(config.DEVICE)
            true_masks = data['masks'].to(config.DEVICE)

            masks = model(images)

            score = smp_dice_score(masks, true_masks)
            score = score.detach().cpu().item()
            dice_scores += score

            loss = loss_fn(masks, true_masks)

            batch_loss = loss.detach().cpu().item()
            total_loss += batch_loss

            loop.set_postfix(dict(
                loss = batch_loss,
                dice_score = score
            ))

    return round(total_loss/len(data_loader), 4), round(dice_scores/len(data_loader), 4)