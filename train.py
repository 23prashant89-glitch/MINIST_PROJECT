import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import Config
from model import MLP
from utils import sefup_derectories, get_logger, set_seed


def get_data_loaders(config):
    """Create MNIST train/validation DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='../data', train=True, download=True, transform=transform
    )
    # Split: 50k train, 10k validation
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [50000, 10000]
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.bacth_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.bacth_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, logger):
    """Run one training epoch, return average loss."""
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            logger.info(
                f'  Batch [{batch_idx+1:>4}/{len(loader)}]  '
                f'Loss: {loss.item():.4f}'
            )

    return running_loss / len(loader)


def validate(model, loader, criterion, device, logger):
    """Run validation, return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, filepath, logger):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': {
            'input_size': Config.input_size,
            'hidden_size': Config.hidden_size,
            'num_classes': Config.num_classes,
            'dropout': Config.dropout,
        }
    }
    torch.save(checkpoint, filepath)
    logger.info(f'Checkpoint saved -> {filepath}')


def main():
    config = Config()

    # Setup
    sefup_derectories(config)
    logger = get_logger(config)
    set_seed(42)
    device = torch.device(config.device)
    logger.info(f'Using device: {device}')
    logger.info(f'Config: batch_size={config.bacth_size}, '
                f'lr={config.learning_rate}, '
                f'epochs={config.max_epoch}, '
                f'hidden={config.hidden_size}')

    # Data
    logger.info('Loading MNIST data ...')
    train_loader, val_loader = get_data_loaders(config)
    logger.info(f'Train batches: {len(train_loader)}, '
                f'Val batches: {len(val_loader)}')

    # Model
    model = MLP(
        input_size=config.input_size,
        hidden_sizes=config.hidden_size,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(device)
    logger.info(f'Model: {model.__class__.__name__} — '
                f'{sum(p.numel() for p in model.parameters()):,} parameters')

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.shedular_patience,
        factor=config.shedular_factor
    )

    # Early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_checkpoint_path = None

    # Training loop
    logger.info('-' * 60)
    logger.info('Starting training ...')
    logger.info('-' * 60)

    for epoch in range(1, config.max_epoch + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, logger)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch summary
        logger.info(
            f'Epoch [{epoch:>2}/{config.max_epoch}]  '
            f'Train Loss: {train_loss:.4f}  '
            f'Val Loss: {val_loss:.4f}  '
            f'Val Acc: {val_acc:.2f}%  '
            f'LR: {current_lr:.2e}'
        )

        # Checkpoint — save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_checkpoint_path = (
                f'{config.checkpoint_dir}/best_model_epoch{epoch}_'
                f'val{val_acc:.2f}.pth'
            )
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                best_checkpoint_path, logger
            )
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= config.early_stop_patience:
            logger.info(
                f'Early stopping triggered after {epoch} epochs '
                f'({epochs_no_improve} epochs without improvement).'
            )
            break

    # Final summary
    logger.info('=' * 60)
    logger.info('Training complete!')
    logger.info(f'Best validation loss: {best_val_loss:.4f}')
    if best_checkpoint_path:
        logger.info(f'Best checkpoint: {best_checkpoint_path}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()