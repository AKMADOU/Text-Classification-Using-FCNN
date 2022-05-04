import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from torch import optim
from model import FeedfowardTextClassifier
from tqdm import tqdm

def train_epoch(model, optimizer, train_loader,criterion, scheduler, input_type='bow'):
    model.train()
    total_loss, total = 0, 0
    for seq, bow, tfidf, target, text in train_loader:
        if input_type == 'bow':
            inputs = bow
        if input_type == 'tfidf':
            inputs = tfidf
        
        # Reset gradient
        optimizer.zero_grad()
        
        # Forward pass
        output = model(inputs)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Perform gradient descent, backwards pass
        loss.backward()

        # Take a step in the right direction
        optimizer.step()
        scheduler.step()

        # Record metrics
        total_loss += loss.item()
        total += len(target)

    return total_loss / total


def validate_epoch(model, valid_loader,criterion, input_type='bow'):
    model.eval()
    total_loss, total = 0, 0
    with torch.no_grad():
        for seq, bow, tfidf, target, text in valid_loader:
            if input_type == 'bow':
                inputs = bow
            if input_type == 'tfidf':
                inputs = tfidf

            # Forward pass
            output = model(inputs)

            # Calculate how wrong the model is
            loss = criterion(output, target)

            # Record metrics
            total_loss += loss.item()
            total += len(target)

    return total_loss / total

def TrainManyEpochs(model,train_loader,input_type, optimizer,valid_loader, criterion,scheduler):
    n_epochs = 0
    train_losses, valid_losses = [], []
    while True:
        train_loss = train_epoch(model, optimizer, train_loader,criterion, scheduler, input_type='bow')
        valid_loss = validate_epoch(model, valid_loader,criterion, input_type='bow')
        
        tqdm.write(
            f'epoch #{n_epochs + 1:3d}\ttrain_loss: {train_loss:.2e}\tvalid_loss: {valid_loss:.2e}\n',
        )
        
        # Early stopping if the current valid_loss is greater than the last three valid losses
        if len(valid_losses) > 2 and all(valid_loss >= loss for loss in valid_losses[-3:]):
            print('Stopping early')
            break
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        n_epochs += 1
    return train_losses, valid_losses, n_epochs