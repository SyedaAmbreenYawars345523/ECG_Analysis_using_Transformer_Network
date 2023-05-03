### train code

#import wandb

# experiment_2_1value_encoder

# matplotlib inline
import matplotlib.pyplot as plt
import time
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
#from natsort import natsorted
from transformer_encoder_decoder import EcgModel
import numpy as np
import os
from pandas import *
from sklearn.model_selection import train_test_split

# Model Parameters
d_model = 64
nhead = 4
num_layers = 4
num_conv_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets  device for model and pytorch tensors

# Training parameters
start_epoch = 0
epochs = 100  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation loss
batch_size = 32
workers = 1  # for data-loading; right now
lr = 1e-4  # learning rate
best_loss = np.inf  # Best score right now for regression infinity
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none


def main():
    """
    Training and validation.

    """

    global batch_size, best_loss, epochs_since_improvement, checkpoint, start_epoch

    ############################
    # getting ecgs and targets :
    ############################

    X, y = get_train_val_data('/content/drive/MyDrive/all_mormal_ecg', '/content/drive/MyDrive/all_mormal_ecg/',
                              '/content/drive/MyDrive/filtered_all_normals_121977_ground_truth.csv')
    print(y.shape)
    #################################
    # train-test split of the dataset
    #################################
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)

    # preparing Dataloader:

    train_ds = TensorDataset(X_train, y_train)

    val_ds = TensorDataset(X_test, y_test)

    if checkpoint is None:
        model = EcgModel(d_model, nhead, num_layers, num_conv_layers)
        # Adam
        model_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                           weight_decay=1e-6)
        # AdamW ((β1=0.9, β2=0.98, ε=10^(-9)))
        # model_optimizer = torch.optim.AdamW(params=filter(lambda p:p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.98), eps=1e-09)


    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epoch_since_improvement']
        best_loss = checkpoint['loss']
        model = checkpoint['model']
        model_optimizer = checkpoint['model_optimizer']

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = nn.MSELoss().to(device)
    mae_loss = nn.L1Loss().to(device)

    # Log the network weight histograms (optional)
    wandb.watch(model)

    # Custom dataloaders
    train_loader = DataLoader(train_ds, batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Epochs
    history_trainval = {
        'train_loss_plot': [],
        'validate_loss_plot': [],
        'train_mae_loss_plot': [],
        'validate_mae_loss_plot': []
        # average score??? do we need to calculate??
    }

    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs and  terminate training after 25.
        if epochs_since_improvement == 25:
            break

        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(model_optimizer, 0.80)

        # One epoch's training
        history = train(train_loader=train_loader,
                        model=model,
                        criterion=criterion,
                        model_optimizer=model_optimizer,
                        epoch=epoch, mae_loss=mae_loss)
        # print(history)

        # One epoch's validation
        recent_loss, history_val = validate(val_loader=val_loader,
                                            model=model,
                                            criterion=criterion, mae_loss=mae_loss)
        # print(history_val)
        history_trainval['train_loss_plot'].append(sum(history['loss']) / len(history['loss']))
        history_trainval['validate_loss_plot'].append(sum(history_val['loss_val']) / len(history_val['loss_val']))
        history_trainval['train_mae_loss_plot'].append(sum(history['mae_loss']) / len(history['mae_loss']))
        history_trainval['validate_mae_loss_plot'].append(sum(history_val['mae_val']) / len(history_val['mae_val']))

        # Check if there was an improvement
        is_best = recent_loss < best_loss

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0
            best_loss = recent_loss

        # Save checkpoint
        #put train file name below
        save_checkpoint("......", epoch, epochs_since_improvement, model, model_optimizer,
                        recent_loss, is_best)

    # 🐝 Close your wandb run
    wandb.finish()

    epo = range(1, (len(history_trainval['train_loss_plot'])) + 1)
    plt.figure(figsize=(10, 10), dpi=80)
    plt.plot(epo, history_trainval['train_loss_plot'], 'g', label='Training MSE loss')
    plt.plot(epo, history_trainval['validate_loss_plot'], 'r', label='Validation MSE loss')
    plt.plot(epo, history_trainval['train_mae_loss_plot'], 'b', label='Training MAE loss')
    plt.plot(epo, history_trainval['validate_mae_loss_plot'], 'y', label='Validation MAE loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(-2, 100)
    plt.legend(loc='upper left')

    plt.show()


def train(train_loader, model, criterion, model_optimizer, epoch, mae_loss):
    """
        Performs one epoch's training.

        :param train_loader: DataLoader for training data
        :param model: ECG1Model model
        :param criterion: MSE loss
        :param model_optimizer: optimizer to update model's weights
        :param epoch: epoch number
        """
    history = {

        'loss': [],
        'mae_loss': []
    }
    model.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    mae_losses = AverageMeter()

    start = time.time()

    # Batches
    for batch, (X, y) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        X = X.to(device)
        y = y.to(device)

        n_queries = torch.zeros_like(y)

        pred = model(X, n_queries)

        # Calculate loss
        loss = criterion(pred, y)

        mae = mae_loss(pred, y)

        model_optimizer.zero_grad()
        loss.backward()

        # Update weights
        model_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)
        mae_losses.update(mae.item())

        start = time.time()

        # Print status
        if batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae.val:.4f} ({mae.avg:.4f})\t'.format(epoch, batch, len(train_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time, loss=losses, mae=mae_losses))
    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))
    print('\n * MAE - {mae.avg:.3f}\n'.format(mae=mae_losses))

    history['loss'].append(losses.avg)
    history['mae_loss'].append(mae_losses.avg)
    wandb.log({'train_loss': losses.avg})
    wandb.log({'train_mae_loss': mae_losses.avg})

    return history


def validate(val_loader, model, criterion, mae_loss):
    """
        Performs one epoch's validation.

        :param val_loader: DataLoader for validation data.
        :param encoder: encoder model
        :param decoder: decoder model
        :param criterion: loss layer
        :return: Average score
        """
    history_val = {
        'loss_val': [],
        'mae_val': []
    }
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_losses = AverageMeter()

    start = time.time()

    with torch.no_grad():
        # Batches
        for batch, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)

            n_queries = torch.zeros_like(y)

            pred = model(X, n_queries)

            loss = criterion(pred, y)
            mae = mae_loss(pred, y)

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)
            mae_losses.update(mae.item())

            start = time.time()

            # Print status
            if batch % 10 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae.val:.4f} ({mae.avg:.4f})\t'.format(batch, len(val_loader), batch_time=batch_time,
                                                                   loss=losses, mae=mae_losses))
        print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))
        print('\n * MAE - {mae.avg:.3f}\n'.format(mae=mae_losses))

        history_val['loss_val'].append(losses.avg)
        history_val['mae_val'].append(mae_losses.avg)
        average_loss = sum(history_val['loss_val']) / len(history_val['loss_val'])
        average_mae = sum(history_val['mae_val']) / len(history_val['mae_val'])
        wandb.log({'val_loss': average_loss})
        wandb.log({'val_mae_loss': average_mae})

        return average_loss, history_val


if __name__ == '__main__':
    main()


