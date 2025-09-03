import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
import time

def train_model(dataset, model_class=None, epochs=100, batch_size=32, lr=0.001, device='auto', 
               early_stopping_patience=15, weight_decay=1e-4):
    """Train brain-to-image decoder with M4 optimizations"""
    
    # Setup device - prioritize Apple Metal Performance Shaders for M4
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple Metal Performance Shaders (MPS) - M4 Optimized!")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA GPU")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device)
    print(f"Training device: {device}")
    
    # Split dataset
    train_size = int(0.85 * len(dataset))  # Use more data for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders with M4 optimizations
    num_workers = 8 if device.type in ['mps', 'cuda'] else 4  # M4 multicore optimization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")
    
    # Initialize model  
    sample_fmri, sample_img, _ = dataset[0]
    n_features = sample_fmri.shape[0]  # This is now PCA-reduced size!
    img_size = sample_img.shape[1]  # Assuming square images
    
    print(f"Model input features: {n_features} (PCA-reduced)")
    
    if model_class is None:
        from simple_decoder import SimpleBrainDecoder
        model_class = SimpleBrainDecoder
    
    model = model_class(n_features, img_size).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Enhanced loss function (MSE + L1)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    def combined_loss(pred, target, mse_weight=0.7, l1_weight=0.3):
        return mse_weight * mse_loss(pred, target) + l1_weight * l1_loss(pred, target)
    
    # Advanced optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop with advanced features
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (fmri_batch, img_batch, _) in enumerate(train_loader):
            fmri_batch = fmri_batch.to(device, non_blocking=True)
            img_batch = img_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_imgs = model(fmri_batch)
            loss = combined_loss(pred_imgs, img_batch)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for fmri_batch, img_batch, _ in val_loader:
                fmri_batch = fmri_batch.to(device, non_blocking=True)
                img_batch = img_batch.to(device, non_blocking=True)
                
                pred_imgs = model(fmri_batch)
                loss = combined_loss(pred_imgs, img_batch)
                val_loss += loss.item()
        
        # Record losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Progress reporting
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 10 == 0 or epoch < 5:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, "
                  f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience: {early_stopping_patience})")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    return model, train_losses, val_losses

def visualize_results(model, dataset, device, n_examples=5):
    """Visualize reconstruction results with enhanced display"""
    model.eval()
    
    fig, axes = plt.subplots(2, n_examples, figsize=(20, 8))
    
    with torch.no_grad():
        for i in range(n_examples):
            fmri, true_img, stim_id = dataset[i]
            
            # Generate prediction
            fmri_batch = fmri.unsqueeze(0).to(device)
            pred_img = model(fmri_batch)[0]
            
            # Convert to numpy for plotting
            true_img_np = true_img.permute(1, 2, 0).cpu().numpy()
            pred_img_np = pred_img.permute(1, 2, 0).cpu().numpy()
            
            # Plot
            axes[0, i].imshow(true_img_np)
            # Handle both string and numeric stim_id
            stim_str = str(stim_id)[:8] if len(str(stim_id)) > 8 else str(stim_id)
            axes[0, i].set_title(f"True: {stim_str}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(pred_img_np)
            axes[1, i].set_title("Predicted")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"reconstruction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()

def save_model(model, filepath):
    """Save trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__
    }, filepath)
    print(f"Model saved to {filepath}")

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()

# Main training execution
if __name__ == "__main__":
    from brain_image_dataset import create_dataset
    from simple_decoder import ImprovedBrainDecoder, AdvancedBrainDecoder
    
    # Create full dataset
    print("Creating optimized dataset...")
    dataset = create_dataset(
        data_dir="~/ds001246-download", 
        subject='sub-01', 
        max_samples=None,  # Use ALL data
        img_size=128  # Higher resolution
    )
    
    print(f"Full dataset size: {len(dataset)}")
    
    # Train optimized model
    print("Starting optimized training...")
    model, train_losses, val_losses = train_model(
        dataset, 
        model_class=ImprovedBrainDecoder,  # Use advanced model
        epochs=150,  # More epochs
        batch_size=32,  # Larger batch size for M4
        lr=0.0005,  # Lower learning rate for stability
        device='auto'  # Auto-detect MPS for M4
    )
    
    # Plot training progress
    plot_training_curves(train_losses, val_losses)
    
    # Visualize results
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    visualize_results(model, dataset, device, n_examples=8)
    
    # Save model
    save_model(model, "optimized_brain_decoder_m4.pth")
    
    print("Optimized training complete!")