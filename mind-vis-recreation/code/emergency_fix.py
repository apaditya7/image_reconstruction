#!/usr/bin/env python3
"""
EMERGENCY FIX: Ultra-fast brain decoder for M4
Uses aggressive PCA to reduce 200k voxels to manageable size
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
from sklearn.decomposition import PCA
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class TinyBrainDecoder(nn.Module):
    """Ultra-small decoder after PCA dimensionality reduction"""
    
    def __init__(self, n_features, img_size=64):
        super().__init__()
        self.img_size = img_size
        
        # Tiny network for PCA-reduced features
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, img_size * img_size * 3),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        pixels = self.net(features)
        batch_size = features.size(0)
        return pixels.view(batch_size, 3, self.img_size, self.img_size)

def emergency_train():
    print("🚨 EMERGENCY FIX - PCA + Tiny Model")
    print("=" * 40)
    
    from brain_image_dataset import create_dataset
    
    # Load small dataset
    print("Loading dataset...")
    dataset = create_dataset('/Users/adityaap/ds001246-download', max_samples=50, img_size=64)
    print(f"Dataset: {len(dataset)} samples")
    
    # Extract all fMRI features for PCA
    print("Extracting fMRI features for PCA...")
    all_fmri = []
    all_images = []
    all_stim_ids = []
    
    for i in range(len(dataset)):
        fmri, img, stim_id = dataset[i]
        all_fmri.append(fmri.numpy())
        all_images.append(img)
        all_stim_ids.append(stim_id)
    
    all_fmri = np.stack(all_fmri)
    print(f"fMRI data shape: {all_fmri.shape}")
    
    # Apply PCA to reduce from ~200k features to 100
    print("Applying PCA dimensionality reduction...")
    pca = PCA(n_components=100)  # Reduce to just 100 features!
    reduced_fmri = pca.fit_transform(all_fmri)
    
    print(f"Reduced from {all_fmri.shape[1]:,} to {reduced_fmri.shape[1]} features")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Create new dataset with reduced features
    class ReducedDataset:
        def __init__(self, fmri_features, images, stim_ids):
            self.fmri_features = torch.FloatTensor(fmri_features)
            self.images = torch.stack(images)
            self.stim_ids = stim_ids
        
        def __len__(self):
            return len(self.fmri_features)
        
        def __getitem__(self, idx):
            return self.fmri_features[idx], self.images[idx], self.stim_ids[idx]
    
    reduced_dataset = ReducedDataset(reduced_fmri, all_images, all_stim_ids)
    
    # Split dataset
    train_size = int(0.8 * len(reduced_dataset))
    val_size = len(reduced_dataset) - train_size
    train_dataset, val_dataset = random_split(reduced_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using MPS!")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU")
    
    # Create tiny model
    model = TinyBrainDecoder(n_features=100, img_size=64).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (MUCH smaller!)")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Fast training
    epochs = 25
    print(f"\nTraining {epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), desc="Fast Training"):
        # Training
        model.train()
        train_loss = 0
        
        for fmri_batch, img_batch, _ in train_loader:
            fmri_batch = fmri_batch.to(device)
            img_batch = img_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(fmri_batch)
            loss = criterion(pred, img_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation  
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for fmri_batch, img_batch, _ in val_loader:
                fmri_batch = fmri_batch.to(device)
                img_batch = img_batch.to(device)
                
                pred = model(fmri_batch)
                loss = criterion(pred, img_batch)
                val_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train={train_loss/len(train_loader):.4f}, Val={val_loss/len(val_loader):.4f}")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training done in {total_time:.1f} seconds!")
    
    # Show results
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    with torch.no_grad():
        for i in range(5):
            if i < len(reduced_dataset):
                fmri, true_img, stim_id = reduced_dataset[i]
                
                pred_img = model(fmri.unsqueeze(0).to(device))[0]
                
                true_np = true_img.permute(1, 2, 0).numpy()
                pred_np = pred_img.permute(1, 2, 0).cpu().numpy()
                
                axes[0, i].imshow(true_np)
                axes[0, i].set_title(f"True: {str(stim_id)[:6]}")
                axes[0, i].axis('off')
                
                axes[1, i].imshow(pred_np)
                axes[1, i].set_title("Predicted")
                axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("emergency_results.png", dpi=150)
    plt.show()
    
    print(f"\n✅ SUCCESS!")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"📊 Model: {total_params:,} parameters")
    print(f"🧠 Features: {all_fmri.shape[1]:,} → {reduced_fmri.shape[1]} (PCA)")
    print(f"📈 PCA variance: {pca.explained_variance_ratio_.sum():.1%}")

if __name__ == "__main__":
    emergency_train()