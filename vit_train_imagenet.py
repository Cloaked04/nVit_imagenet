import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import time
import os
import wandb
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)


import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

"""
ViT class
"""

# Helper function
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Attention class
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        dropout=0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head  # Added to store dim_head
        self.scale = dim_head ** -0.5  # Scaling factor for query

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, N, 3, self.heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1))

        # Apply softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute output
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# FeedForward class
class FeedForward(nn.Module):
    def __init__(self, dim, dim_inner, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Vision Transformer (ViT) class
class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout=0.0,
        emb_dropout=0.0,
        channels=3,
        dim_head=64
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_size = patch_size
        self.dim = dim

        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_height, pw=patch_width),
            nn.Linear(patch_dim, dim)
        )

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim, dim_head=dim_head, heads=heads, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Patch embedding
        x = self.to_patch_embedding(img)
        B, N, _ = x.shape

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embedding[:, :N + 1, :]
        x = self.dropout(x)

        # Transformer blocks
        for norm1, attn, norm2, ff in self.transformer:
            x = x + attn(norm1(x))
            x = x + ff(norm2(x))

        # Classification head
        x = x[:, 0]
        x = self.mlp_head(x)
        return x

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Initialize wandb
wandb.init(project='vit-imagenet', config={
    'model': 'ViT',
    'dataset': 'ImageNet',
    'epochs': 100,
    'batch_size': 256,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'image_size': 224,
    'patch_size': 16,
    'dim': 768,
    'depth': 12,
    'heads': 12,
    'mlp_dim': 3072,
    'dropout': 0.1,
    'emb_dropout': 0.1,
    'num_classes': 1000
})
config = wandb.config

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(config.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize(config.image_size + 32),
    transforms.CenterCrop(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    ),
])

# Load datasets
train_dataset = datasets.ImageNet(root='datasets/imagenet/train', split='train', transform=train_transforms)
val_dataset = datasets.ImageNet(root='datasets/imagenet/val', split='val', transform=val_transforms)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Initialize the model
model = ViT(
    image_size=config.image_size,
    patch_size=config.patch_size,
    num_classes=config.num_classes,
    dim=config.dim,
    depth=config.depth,
    heads=config.heads,
    mlp_dim=config.mlp_dim,
    dropout=config.dropout,
    emb_dropout=config.emb_dropout
).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# Learning rate scheduler (optional)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# Function to save checkpoint
def save_checkpoint(state, checkpoint_dir='checkpoints', filename='latest_checkpoint.pth.tar'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

# Function to load checkpoint (if resuming training)
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    return start_epoch

# Function to compute gradient norms
def compute_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

# Register hooks for activation statistics
activation_stats = {}

def get_activation(name):
    def hook(model, input, output):
        activation_stats[name] = output.detach()
    return hook

# Example: Register hooks on the first transformer block
model.transformer[0].attn.attn.register_forward_hook(get_activation('attn_output'))
model.transformer[0].ff.net[0].register_forward_hook(get_activation('ff_output'))

# Training loop with checkpointing every 20 hours
start_time = time.time()
checkpoint_interval = 20 * 3600  # 20 hours in seconds
last_checkpoint_time = start_time

num_epochs = config.epochs
best_val_accuracy = 0.0

for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += images.size(0)

        # Compute gradient norm
        grad_norm = compute_gradient_norm(model)

        # Log metrics every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                'Train Loss': running_loss / total_samples,
                'Train Accuracy': total_correct / total_samples,
                'Learning Rate': optimizer.param_groups[0]['lr'],
                'Epoch': epoch + batch_idx / len(train_loader),
                'Gradient Norm': grad_norm,
            })

        # Log weight and gradient histograms every 500 batches
        if batch_idx % 500 == 0:
            for name, param in model.named_parameters():
                wandb.log({f"Weights/{name}": wandb.Histogram(param.data.cpu())})
                if param.grad is not None:
                    wandb.log({f"Gradients/{name}": wandb.Histogram(param.grad.data.cpu())})

        # Check if it's time to save a checkpoint
        elapsed_time = time.time() - last_checkpoint_time
        if elapsed_time >= checkpoint_interval:
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_state)
            last_checkpoint_time = time.time()  # Reset checkpoint timer

    # Scheduler step
    scheduler.step()

    # Training time per epoch
    epoch_duration = time.time() - epoch_start_time
    wandb.log({'Epoch Duration': epoch_duration})

    # GPU memory usage
    gpu_memory_allocated = torch.cuda.memory_allocated(device)
    gpu_memory_reserved = torch.cuda.memory_reserved(device)
    wandb.log({
        'GPU Memory Allocated': gpu_memory_allocated,
        'GPU Memory Reserved': gpu_memory_reserved
    })

    # Activation statistics
    for name, activation in activation_stats.items():
        wandb.log({
            f'Activations/{name}_Mean': activation.mean().item(),
            f'Activations/{name}_Std': activation.std().item()
        })

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # For per-class accuracy
    class_correct = [0] * config.num_classes
    class_total = [0] * config.num_classes

    # For sample images with predictions
    sample_images = []
    sample_captions = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += images.size(0)

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

            # Collect sample images and predictions
            if batch_idx == 0:
                images_cpu = images.cpu()
                preds_cpu = predicted.cpu()
                labels_cpu = labels.cpu()
                for img, pred, label in zip(images_cpu, preds_cpu, labels_cpu):
                    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + \
                          torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)  # Unnormalize
                    img = torch.clamp(img, 0, 1)
                    sample_images.append(wandb.Image(img, caption=f"Pred: {pred}, Label: {label}"))
                # Limit to 32 images
                if len(sample_images) >= 32:
                    sample_images = sample_images[:32]
                    break

    avg_train_loss = running_loss / total_samples
    avg_train_accuracy = total_correct / total_samples
    avg_val_loss = val_loss / val_total
    avg_val_accuracy = val_correct / val_total

    # Log epoch metrics to wandb
    wandb.log({
        'Epoch': epoch,
        'Train Loss': avg_train_loss,
        'Train Accuracy': avg_train_accuracy,
        'Validation Loss': avg_val_loss,
        'Validation Accuracy': avg_val_accuracy,
        'Learning Rate': optimizer.param_groups[0]['lr']
    })

    # # Log per-class accuracy
    # for i in range(config.num_classes):
    #     if class_total[i] > 0:
    #         wandb.log({f'Class Accuracy/Class_{i}': class_correct[i] / class_total[i]})

    # # Log sample images
    # if sample_images:
    #     wandb.log({'Sample Images': sample_images})

    print(f'Epoch [{epoch}/{num_epochs}] '
          f'Train Loss: {avg_train_loss:.4f} '
          f'Train Acc: {avg_train_accuracy:.4f} '
          f'Val Loss: {avg_val_loss:.4f} '
          f'Val Acc: {avg_val_accuracy:.4f}')

    # Save best model
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        checkpoint_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint_state, filename='best_model.pth.tar')

# Save final model
checkpoint_state = {
    'epoch': num_epochs,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
save_checkpoint(checkpoint_state, filename='final_model.pth.tar')

# Finish wandb run
wandb.finish()