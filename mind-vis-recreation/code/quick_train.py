#!/usr/bin/env python3
"""
QUICK FIX: Fast brain-to-image training for M4 MacBook Pro
Designed to train in under 5 minutes with reasonable results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class QuickBrainDecoder(nn.Module):
    """Fast decoder optimized for quick training"""
    
    def __init__(self, n_voxels, img_size=64):
        super().__init__()
        self.img_size = img_size
        
        # Fast encoder with aggressive dimension reduction
        self.encoder = nn.Sequential(
            nn.Linear(n_voxels, 256),  # Huge reduction
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Very compact
        )
        
        # Simple decoder
        n_pixels = img_size * img_size * 3
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, n_pixels),
            nn.Sigmoid()
        )
    
    def forward(self, fmri_features):
        encoded = self.encoder(fmri_features)
        pixels = self.decoder(encoded)
        
        batch_size = fmri_features.size(0)
        images = pixels.view(batch_size, 3, self.img_size, self.img_size)
        
        return images

def quick_train():
    print("🚀 QUICK TRAINING MODE - Optimized for M4 MacBook Pro")
    print("=" * 50)
    
    # Import dataset
    from brain_image_dataset import create_dataset
    
    # Quick configuration
    DATA_DIR = os.path.expanduser("~/ds001246-download")
    
    print("Loading small dataset for quick training...")
    dataset = create_dataset(
        data_dir=DATA_DIR,
        subject='sub-01',
        max_samples=200,  # Small dataset
        img_size=64       # Small images
    )
    
    print(f"Dataset: {len(dataset)} samples")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Quick dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using MPS acceleration!")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU (MPS not available)")
    
    # Initialize model
    sample_fmri, sample_img, _ = dataset[0]
    n_voxels = sample_fmri.shape[0]
    img_size = sample_img.shape[1]
    
    model = QuickBrainDecoder(n_voxels, img_size).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training
    epochs = 20  # Just 20 epochs for speed
    print(f"\nTraining for {epochs} epochs...")
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training
        model.train()
        train_loss = 0.0
        
        for fmri_batch, img_batch, _ in train_loader:
            fmri_batch = fmri_batch.to(device, non_blocking=True)
            img_batch = img_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred_imgs = model(fmri_batch)
            loss = criterion(pred_imgs, img_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for fmri_batch, img_batch, _ in val_loader:
                fmri_batch = fmri_batch.to(device, non_blocking=True)
                img_batch = img_batch.to(device, non_blocking=True)
                
                pred_imgs = model(fmri_batch)
                loss = criterion(pred_imgs, img_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time:.1f} seconds!")
    
    # Quick visualization
    print("Generating results...")
    model.eval()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    with torch.no_grad():
        for i in range(5):
            fmri, true_img, stim_id = dataset[i]
            
            fmri_batch = fmri.unsqueeze(0).to(device)
            pred_img = model(fmri_batch)[0]
            
            true_img_np = true_img.permute(1, 2, 0).cpu().numpy()
            pred_img_np = pred_img.permute(1, 2, 0).cpu().numpy()
            
            axes[0, i].imshow(true_img_np)
            axes[0, i].set_title(f"True: {str(stim_id)[:8]}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(pred_img_np)
            axes[1, i].set_title("Predicted")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("quick_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save model
    torch.save(model.state_dict(), "quick_brain_decoder.pth")
    
    print(f"\n✅ SUCCESS!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📊 Model size: {total_params:,} parameters")
    print(f"💾 Saved: quick_brain_decoder.pth, quick_results.png")

if __name__ == "__main__":
    quick_train()