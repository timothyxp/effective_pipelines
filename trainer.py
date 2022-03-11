import numpy as np
from wandb_logger import WanDBWriter
from tqdm import tqdm
import torch
from utils import count_zero_grads


def train_block1(
    train_loader, model, criterion, opt, epoch, num_epochs,
    device, scaler, config, wandb_logger: WanDBWriter=None
):
    model.train()
    accs = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        if wandb_logger is not None:
            wandb_logger.set_step(wandb_logger.step + 1, mode='train')
            
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        accuracy = ((outputs > 0.5) == labels).float().mean() \
            .detach().cpu().numpy()
        np_loss = loss.detach().cpu().numpy()
        accs.append(accuracy)

        pbar.set_description(
            f"Loss: {np_loss:.4f}"
            f"Accuracy: {accuracy:.4f}"
            f"Epoch acc: {np.mean(accs) * 100:.4f}"
        )
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        
        if wandb_logger is not None:
            wandb_logger.add_scalar("loss", np_loss)
            wandb_logger.add_scalar("accuracy", accuracy)
            wandb_logger.add_scalar("zero_grads_rate", count_zero_grads(opt))
        
        scaler.step(opt)
        scaler.update()
        if wandb_logger is not None and hasattr(scaler, 'scale_rate'):
            wandb_logger.add_scalar('scale_rate', scaler.scale_rate)
        
        opt.zero_grad()
        
    if wandb_logger is not None:
        wandb_logger.add_scalar("epoch_accuracy", np.mean(accs))
        
        

def train_block2(
    train_loader, model, device, config, wandb_logger: WanDBWriter=None
):
    model.train()
    accs = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in pbar:
        if wandb_logger is not None:
            wandb_logger.set_step(wandb_logger.step + 1, mode='train')
            
        data = data.to(device, non_blocking=True)
        
        src_mask = torch.zeros((data.shape[0], data.shape[0]), device=device)
        out = model(data, src_mask).mean().item()
    
        if wandb_logger is not None:
            wandb_logger.add_scalar("out", out)
