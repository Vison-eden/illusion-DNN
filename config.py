from __future__ import absolute_import, division, print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.data_loader import TrainDataset,TestDataset
from models.models import Brainmodels
from data.data_loader import transform_train, transform_test

#Define model name
models_name = ["resnext101_32x8d", "efficientnet-b1", "resnet152", "pnasnet5large",
               "efficientnet-b6", "resnet101", "densenet201", "vgg19"]

# Set command line parameters
parser = argparse.ArgumentParser(description='Illusion in DNN')
parser.add_argument('--m_name', type=str, default=models_name[5], choices=models_name, help="The model selection")
parser.add_argument('--lr', type=float, default=0.001, help="The learning rate for training")
parser.add_argument('--epoch', type=int, default=0, help="The number of epochs on model's training")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
parser.add_argument('--checkpoint_path', type=str, default="checkpoints", help="Path to save the model checkpoints")
parser.add_argument('--train_path', type=str, default="./dataset/train", help="Path to the training dataset")
parser.add_argument('--val_path', type=str, default="./dataset/value", help="Path to the validation dataset")
parser.add_argument('--workers', type=int, default=8, help="Number of workers for data loading")
parser.add_argument('--num_epoch', type=int, default=100, help="Total number of training epochs")
parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay for optimization")
parser.add_argument('--pretrained', type=bool, default=True, help="Use pretrained model")
parser.add_argument('--classes', type=int, default=2, help="Number of classes in the dataset")
args = parser.parse_args()

def make_args(setting):
    return setting

# Helper functions
def adjust_learning_rate(optimizer, epoch, init_lr):
    """Adjusts the learning rate according to epoch number."""
    lr = init_lr * (0.9 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate_accuracy(data_iter, net, device=None):
    """Evaluates accuracy of the model on a given dataset."""
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for (input_te, label_te) in data_iter:
            net.eval()
            acc_sum += (net(input_te.to(device)).argmax(dim=1) == label_te.to(device)).float().sum().cpu().item()
            n += label_te.shape[0]
    return acc_sum / n


# Training function
def train(config):
    # Setup model saving directory
    model_save_dir = os.path.join("./models", config.m_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # Load datasets
    train_data = TrainDataset(data_dir=config.train_path, transform=transform_train)
    val_data = TrainDataset(data_dir=config.val_path, transform=transform_train)
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.workers)
    val_loader = DataLoader(dataset=val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training model: {config.m_name} on device: {device}')

    # Model setup
    net = Brainmodels(config.m_name, config.classes, config.pretrained).final_models()
    if config.classes != 1000:  # Assuming 1000 is the default class number for pretrained models
        # Replace the last layer if the number of classes is different
        net.fc = nn.Sequential(
            nn.Linear(2048, 10),
            nn.ReLU(),
            nn.Linear(10, config.classes),
            nn.LogSoftmax(dim=1)
        )
    model = net.to(device)

    # Optimizer and loss function
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Training loop
    for epoch in tqdm(range(config.num_epoch), desc="Training"):
        adjust_learning_rate(optimizer, epoch, config.lr)
        # Training phase
        model.train()
        train_loss, train_acc, total = 0.0, 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        avg_train_acc = train_acc / total

        # Validation phase
        model.eval()
        val_loss, val_acc, total = 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / total
        avg_val_acc = val_acc / total

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{config.num_epoch}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')

        # Save model
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(model_save_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    con_info = make_args(args)
    train(con_info)
