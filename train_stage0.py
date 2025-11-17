#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the model architecture
import sys
sys.path.append('../Kaggle_ECGnet')
from stage0_model import Net

class ECGDataset(Dataset):
    """Dataset for ECG image training"""

    def __init__(self, data_dir, csv_file, transform=None, mode='train'):
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode

        # Create list of all image paths
        self.samples = []
        for idx, row in self.df.iterrows():
            image_id = str(row['id'])
            # Get all available image types for this image
            image_path = os.path.join(data_dir, 'train', image_id)
            if os.path.exists(image_path):
                for type_id in ['0001', '0003', '0004', '0005', '0006', '0009', '0010', '0011', '0012']:
                    img_file = os.path.join(image_path, f'{image_id}-{type_id}.png')
                    if os.path.exists(img_file):
                        self.samples.append({
                            'image_path': img_file,
                            'image_id': image_id,
                            'type_id': type_id,
                            'sig_len': row.get('sig_len', 5000)
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['image_path'], cv2.IMREAD_COLOR_RGB)
        if image is None:
            # Fallback to dummy image
            image = np.zeros((960, 1280, 3), dtype=np.uint8)

        # Generate dummy training labels (in real scenario, these would come from annotations)
        H, W = image.shape[:2]
        marker = np.random.randint(0, 14, (H, W), dtype=np.int64)  # 13 markers + background
        orientation = np.random.randint(0, 8, dtype=np.int64)  # 8 orientations

        if self.transform:
            augmented = self.transform(image=image, marker=marker)
            image = augmented['image']
            marker = augmented['marker']

        return {
            'image': torch.from_numpy(image).byte(),
            'marker': torch.from_numpy(marker).byte(),
            'orientation': torch.from_numpy(np.array([orientation])).byte(),
            'image_id': sample['image_id'],
            'type_id': sample['type_id']
        }

def create_transforms(image_size=(960, 1280)):
    """Create data augmentation transforms"""
    train_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform

def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_marker_loss = 0
    total_orientation_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move data to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast():
            output = model(batch)
            loss = output['marker_loss'] + output['orientation_loss']

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_marker_loss += output['marker_loss'].item()
        total_orientation_loss += output['orientation_loss'].item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'marker': f'{output["marker_loss"].item():.4f}',
            'orient': f'{output["orientation_loss"].item():.4f}'
        })

    return {
        'total_loss': total_loss / len(dataloader),
        'marker_loss': total_marker_loss / len(dataloader),
        'orientation_loss': total_orientation_loss / len(dataloader)
    }

def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_marker_loss = 0
    total_orientation_loss = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            with torch.cuda.amp.autocast():
                output = model(batch)
                loss = output['marker_loss'] + output['orientation_loss']

            # Update metrics
            total_loss += loss.item()
            total_marker_loss += output['marker_loss'].item()
            total_orientation_loss += output['orientation_loss'].item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'marker': f'{output["marker_loss"].item():.4f}',
                'orient': f'{output["orientation_loss"].item():.4f}'
            })

    return {
        'total_loss': total_loss / len(dataloader),
        'marker_loss': total_marker_loss / len(dataloader),
        'orientation_loss': total_orientation_loss / len(dataloader)
    }

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'stage0-epoch-{epoch:08d}.checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)

    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'stage0-last.checkpoint.pth')
    torch.save(checkpoint, latest_path)

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'stage0-best.checkpoint.pth')
        torch.save(checkpoint, best_path)

    return checkpoint_path

def main():
    # Configuration
    config = {
        'data_dir': KAGGLE_DIR,
        'train_csv': f'{KAGGLE_DIR}/train.csv',
        'checkpoint_dir': f'{OUT_DIR}/checkpoints',
        'batch_size': 4,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'device': DEVICE,
        'num_workers': 2,
        'pin_memory': True,
        'mixed_precision': True,
        'image_size': (960, 1280),
        'val_split': 0.2,
        'save_interval': 5
    }

    print("Stage0 Training Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Create transforms
    train_transform, val_transform = create_transforms(config['image_size'])

    # Create datasets
    full_dataset = ECGDataset(config['data_dir'], config['train_csv'], transform=None)

    # Limit dataset size for demo purposes
    if len(full_dataset) > 1000:
        full_dataset.samples = full_dataset.samples[:1000]

    # Split into train/val
    train_size = int((1 - config['val_split']) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    print(f"Dataset size: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    # Create model
    model = Net(pretrained=True)
    model = model.to(config['device'])

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    # Create scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("Starting Stage0 training...")
    print(f"Device: {config['device']}")
    print(f"Mixed precision: {config['mixed_precision']}")
    print()

    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 50)

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, config['device'], scaler)

        # Validate
        val_metrics = validate(model, val_loader, config['device'])

        # Update learning rate
        scheduler.step()

        # Record losses
        train_losses.append(train_metrics['total_loss'])
        val_losses.append(val_metrics['total_loss'])

        # Print metrics
        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(Marker: {train_metrics['marker_loss']:.4f}, "
              f"Orientation: {train_metrics['orientation_loss']:.4f})")
        print(f"Val Loss: {val_metrics['total_loss']:.4f} "
              f"(Marker: {val_metrics['marker_loss']:.4f}, "
              f"Orientation: {val_metrics['orientation_loss']:.4f})")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']

        if (epoch + 1) % config['save_interval'] == 0 or is_best:
            save_checkpoint(model, optimizer, epoch + 1, val_metrics['total_loss'],
                          config['checkpoint_dir'], is_best)

        print()

    # Save final model
    save_checkpoint(model, optimizer, config['num_epochs'], best_val_loss,
                  config['checkpoint_dir'], is_best=True)

    print("Stage0 training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Stage0 Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{OUT_DIR}/stage0_training_curves.png')
    plt.show()

if __name__ == '__main__':
    main()