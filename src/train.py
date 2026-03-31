import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import numpy as np

from dataset import load_metadata, HAM10000Dataset, get_transforms, get_class_weights
from model import build_model, get_device, count_parameters


def get_args():
    parser = argparse.ArgumentParser(description='Train skin lesion classifier')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--freeze-epochs', type=int, default=3)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--model-out', type=str, default='models/best_model.pth')
    parser.add_argument('--experiment-name', type=str, default='skin-lesion-v1')
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def main():
    args = get_args()
    device = get_device()
    print(f'Using device: {device}')

    image_dir = os.path.join(args.data_dir, 'images')
    csv_path = os.path.join(args.data_dir, 'GroundTruth.csv')

    df = load_metadata(csv_path)

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['dx'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['dx'], random_state=42)

    test_df.to_csv(os.path.join(args.data_dir, 'test_split.csv'), index=False)

    label_encoder = LabelEncoder()
    label_encoder.fit(df['dx'])

    train_dataset = HAM10000Dataset(train_df, [image_dir], get_transforms('train'), label_encoder)
    val_dataset = HAM10000Dataset(val_df, [image_dir], get_transforms('val'), label_encoder)

    class_weights = get_class_weights(train_dataset)
    sample_weights = class_weights[train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    pin = device == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    model = build_model(num_classes=7, freeze_backbone=True).to(device)
    print('\nModel parameters (frozen backbone):')
    count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mlflow.set_experiment(args.experiment_name)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'freeze_epochs': args.freeze_epochs,
            'architecture': 'efficientnet_b0',
        })

        for epoch in range(1, args.epochs + 1):

            if epoch == args.freeze_epochs + 1:
                print('\nUnfreezing backbone...')
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.freeze_epochs)
                print('Model parameters (full network):')
                count_parameters(model)

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': current_lr,
            }, step=epoch)

            print(
                f'Epoch {epoch:>2}/{args.epochs} | '
                f'Train loss: {train_loss:.4f} | '
                f'Val loss: {val_loss:.4f} | '
                f'Val acc: {val_acc * 100:.2f}%'
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.model_out)
                print(f'  -> Saved best model (val acc: {best_val_acc * 100:.2f}%)')

    print(f'\nTraining complete. Best val accuracy: {best_val_acc * 100:.2f}%')
    print(f'Model saved to: {args.model_out}')


if __name__ == '__main__':
    main()
