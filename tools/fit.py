import os
import random

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def evaluate(model, criterion, val_dataloader, device):
    val_loss = [0] * 6
    model.eval()
    for batch in tqdm(val_dataloader, desc=f'Evaluation', leave=False):
        images, targets = batch['image'], batch['target']
        images, targets = images.to(device), targets.to(device)

        with torch.no_grad():
            predictions = model(images)
            loss = criterion(predictions, targets)
            val_loss = [x.item() + y for x, y in zip(loss, val_loss)]

    return [item / len(val_dataloader) for item in val_loss]


def fit(model, optimizer, scheduler, criterion, epochs, train_dataloader, val_dataloader,
        train_dataset=None, backup='backup/', verbose=False):
    """
    Training model with drawing graph of the loss curve.

    param: model - model to fitting
    param: optimizer - optimizer loss function
    param: scheduler - optimizer scheduler
    param: criterion - loss function
    param: epochs - number of epochs
    param: train_dataloader - dataloader with training split of dataset
    param: val_dataloader - dataloader with validation split of dataset
    param: train_dataset - train dataset for multiscaling training (default = None, which means multiscale is disabled)
    param: backup - path to save loss graph (default = 'backup/')
    param: verbose - details of loss and resolution (default = False)
    """

    device = model.device

    # create a directory to save the graph if the directory doesn't exist
    if not os.path.isdir(backup):
        os.mkdir(backup)

    train_loss_log = []
    val_loss_log = []
    fig = plt.figure(figsize=(11, 7))
    fig_number = fig.number

    if train_dataset and verbose:
        print('Resolution set to', '[', train_dataset.s * 32, 'x', train_dataset.s * 32, ']')

    for epoch in range(epochs):
        model.train()
        train_loss = [0] * 6

        for item, batch in enumerate(tqdm(train_dataloader, desc=f"Training, epoch {epoch}", leave=False)):
            images, targets = batch['image'], batch['target']
            images, targets = images.to(device), targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)
            train_loss = [x.item() + y for x, y in zip(loss, train_loss)]

            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()

            # every 10 iteration generate a new image scale
            if train_dataset and item % 10 == 9:
                train_dataset.s = random.randint(10, 19)
                if train_dataset and verbose:
                    print('\nResolution change to', '[', train_dataset.s * 32, 'x', train_dataset.s * 32, ']')
            item += 1

        scheduler.step()

        train_loss = [item / len(train_dataloader) for item in train_loss]

        train_loss_log.append(train_loss[0])
        val_loss = evaluate(model, criterion, val_dataloader, device)
        val_loss_log.append(val_loss[0])

        if not plt.fignum_exists(num=fig_number):
            fig = plt.figure(figsize=(11, 7))
            fig_number = fig.number

        print(f"\nepoch: {epoch}\n")
        if verbose:
            print('train:')
            print('coordinate x, y loss:', train_loss[1])
            print('coordinate w, h loss:', train_loss[2])
            print('object loss:', train_loss[3])
            print('no object loss:', train_loss[4])
            print('classes loss:', train_loss[5])
            print('total loss:', train_loss[0])

            print('\nvalidation:')
            print('coordinate x, y loss:', val_loss[1])
            print('coordinate w, h loss:', val_loss[2])
            print('object loss:', val_loss[3])
            print('no object loss:', val_loss[4])
            print('classes loss:', val_loss[5])
            print('total loss:', val_loss[0])
        else:
            print(f"train loss: {train_loss}")
            print(f"val loss: {val_loss}")

        line_train, = plt.plot(list(range(0, epoch + 1)), train_loss_log, color='blue')
        line_val, = plt.plot(list(range(0, epoch + 1)), val_loss_log, color='orange')       
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.ylim([0, 7])
        plt.title("Train steps")
        plt.legend((line_train, line_val), ['train loss', 'validation loss'])
        plt.draw()
        plt.pause(0.001)
        fig.savefig(backup + 'loss.png', bbox_inches='tight')

