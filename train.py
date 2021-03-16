from time import time as now

import progressbar
from torch.optim.lr_scheduler import OneCycleLR
import torch
from torch.cuda.amp import autocast

def epoch(epoch, model, loss_fn, optimizer, scheduler, train_dataloader, valid_dataloader):
    # train
    len_train_dataset = len(train_dataloader.dataset)
    len_valid_dataset = len(valid_dataloader.dataset)

    begin = now()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    idx = 0
    scaler = torch.cuda.amp.GradScaler()
    with progressbar.ProgressBar(max_value=len_train_dataset) as bar:
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                with autocast():
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            idx = idx + inputs.shape[0]
            bar.update(idx)

    epoch_training_loss = running_loss / len_train_dataset

    print(f'TRAINING: LOSS = {epoch_training_loss}')

    # valid
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in valid_dataloader:
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len_valid_dataset
    epoch_accuracy = running_corrects.double() / len_valid_dataset
    print(f'VALID: LOSS = {epoch_loss} ACCURACY = {epoch_accuracy} [{begin - now()}]')
    return (epoch_training_loss, epoch_loss, epoch_accuracy.to('cpu'))


def fit(epochs, model, loss_fn, train_dataloader, valid_dataloader, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.06,
                                                    epochs=epochs,
                                                    steps_per_epoch=len(train_dataloader))
    for i in range(epochs):
        epoch(i, model, loss_fn, optimizer, scheduler, train_dataloader, valid_dataloader)
