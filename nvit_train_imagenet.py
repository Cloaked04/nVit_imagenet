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
import torch.nn.utils.parametrize as parametrize
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.nn import Module, ModuleList

# Helper functions
def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim, p=2)

# For use with parametrize
class L2Norm(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return l2norm(t, dim=self.dim)

class NormLinear(nn.Module):
    def __init__(self, dim, dim_out, norm_dim_in=True):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias=False)

        parametrize.register_parametrization(
            self.linear,
            'weight',
            L2Norm(dim=-1 if norm_dim_in else 0)
        )

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x)

# Attention and FeedForward classes
class Attention(nn.Module):
    def __init__(self, dim, *, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        dim_inner = dim_head * heads
        self.to_q = NormLinear(dim, dim_inner)
        self.to_k = NormLinear(dim, dim_inner)
        self.to_v = NormLinear(dim, dim_inner)

        self.dropout = dropout

        self.q_scale = nn.Parameter(torch.ones(heads, 1, dim_head) * (dim_head ** 0.25))
        self.k_scale = nn.Parameter(torch.ones(heads, 1, dim_head) * (dim_head ** 0.25))

        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(self.split_heads, (q, k, v))

        # Query key rmsnorm
        q, k = map(l2norm, (q, k))

        q = q * self.q_scale
        k = k * self.k_scale

        # Scale is 1., as scaling factor is moved to s_qk (dk ^ 0.25)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.,
            scale=1.
        )

        out = self.merge_heads(out)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, *, dim_inner, dropout=0.):
        super().__init__()
        dim_inner = int(dim_inner * 2 / 3)

        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = NormLinear(dim, dim_inner)
        self.to_gate = NormLinear(dim, dim_inner)

        self.hidden_scale = nn.Parameter(torch.ones(dim_inner))
        self.gate_scale = nn.Parameter(torch.ones(dim_inner))

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in=False)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)

        hidden = hidden * self.hidden_scale
        gate = gate * self.gate_scale * (self.dim ** 0.5)

        hidden = F.silu(gate) * hidden

        hidden = self.dropout(hidden)
        return self.to_out(hidden)

# nViT class
class nViT(nn.Module):
    

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
        dropout=0.,
        channels=3,
        dim_head=64,
        residual_lerp_scale_init=None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        # Calculate patching related stuff
        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)
        num_patches = patch_height_dim * patch_width_dim

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_size, p2=patch_size),
            NormLinear(patch_dim, dim, norm_dim_in=False),
        )

        self.abs_pos_emb = NormLinear(dim, num_patches)

        residual_lerp_scale_init = default(residual_lerp_scale_init, 1. / depth)

        # Layers
        self.dim = dim
        self.scale = dim ** 0.5

        self.layers = nn.ModuleList([])
        self.residual_lerp_scales = nn.ParameterList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, dim_inner=mlp_dim, dropout=dropout),
            ]))

            self.residual_lerp_scales.append(nn.ParameterList([
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init / self.scale),
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init / self.scale),
            ]))

        self.logit_scale = nn.Parameter(torch.ones(num_classes))

        self.to_pred = NormLinear(dim, num_classes)

    @torch.no_grad()
    def norm_weights_(self):
        for module in self.modules():
            if not isinstance(module, NormLinear):
                continue

            normed = module.weight
            original = module.linear.parametrizations.weight.original

            original.copy_(normed)

    def forward(self, images):
        device = images.device

        tokens = self.to_patch_embedding(images)

        seq_len = tokens.shape[-2]
        pos_emb = self.abs_pos_emb.weight[torch.arange(seq_len, device=device)]

        tokens = l2norm(tokens + pos_emb)

        for (attn, ff), (attn_alpha, ff_alpha) in zip(self.layers, self.residual_lerp_scales):

            attn_out = l2norm(attn(tokens))
            tokens = l2norm(tokens.lerp(attn_out, attn_alpha * self.scale))

            ff_out = l2norm(ff(tokens))
            tokens = l2norm(tokens.lerp(ff_out, ff_alpha * self.scale))

        pooled = reduce(tokens, 'b n d -> b d', 'mean')

        logits = self.to_pred(pooled)
        logits = logits * self.logit_scale * self.scale

        return logits


# Initialize wandb
wandb.init(project='nvit-imagenet', config={
    'model': 'nViT',
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
    'num_classes': 1000,
    'dim_head': 64
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

# Define checkpoint path
checkpoint_path = 'checkpoints/latest_checkpoint.pth.tar'

# Initialize the model
model = nViT(
    image_size=config.image_size,
    patch_size=config.patch_size,
    num_classes=config.num_classes,
    dim=config.dim,
    depth=config.depth,
    heads=config.heads,
    mlp_dim=config.mlp_dim,
    dropout=config.dropout,
    dim_head=config.dim_head
).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# Learning rate scheduler (optional)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# Function to load checkpoint (if resuming training)
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
    start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
    return start_epoch


# Training loop with checkpointing every 20 hours
start_time = time.time()
checkpoint_interval = 20 * 3600  # 20 hours in seconds
last_checkpoint_time = start_time

num_epochs = config.epochs
best_val_accuracy = 0.0

# Load checkpoint if available
start_epoch = 1
if os.path.exists(checkpoint_path):
    print(f"Checkpoint found at '{checkpoint_path}'. Resuming training...")
    start_epoch = load_checkpoint(checkpoint_path)
    print(f"Resumed training from epoch {start_epoch}.")



for epoch in range(start_epoch, num_epochs + 1):
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

        # Log metrics every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                'Train Loss': running_loss / total_samples,
                'Train Accuracy': total_correct / total_samples,
                'Learning Rate': optimizer.param_groups[0]['lr'],
                'Epoch': epoch + batch_idx / len(train_loader),
            })

        # Check if it's time to save a checkpoint
        elapsed_time = time.time() - last_checkpoint_time
        if elapsed_time >= checkpoint_interval:
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_val_accuracy': best_val_accuracy
            }
            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(checkpoint_state, f'checkpoints/nvit_checkpoint_epoch_{epoch}.pth.tar')
            last_checkpoint_time = time.time()  # Reset checkpoint timer

    # Scheduler step
    scheduler.step()

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += images.size(0)

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
        # Save best model
        torch.save(checkpoint_state, 'checkpoints/nvit_best_model.pth.tar')

# Save final model
checkpoint_state = {
    'epoch': num_epochs,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(checkpoint_state, 'checkpoints/nvit_final_model.pth.tar')

# Finish wandb run
wandb.finish()