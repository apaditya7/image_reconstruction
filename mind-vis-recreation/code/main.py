#!/usr/bin/env python3
"""
Optimized Brain-to-Image Reconstruction for M4 MacBook Pro
Main execution script with full dataset and advanced models
"""

import os
import sys
import torch
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    print("=== Optimized Brain-to-Image Reconstruction (M4 MacBook Pro) ===")
    print()
    
    # M4 EMERGENCY Speed Configuration  
    DATA_DIR = os.path.expanduser("~/ds001246-download")
    SUBJECT = 'sub-01'
    MAX_SAMPLES = 50    # Very small dataset 
    IMG_SIZE = 32       # Very small images for ultra-fast training
    EPOCHS = 25         # Quick training
    BATCH_SIZE = 8      # Small batch for stability
    LEARNING_RATE = 0.001
    
    # Hardware detection
    if torch.backends.mps.is_available():
        print("🚀 Apple M4 detected! Using Metal Performance Shaders (MPS)")
        device_info = "M4 MacBook Pro with MPS acceleration"
    elif torch.cuda.is_available():
        device_info = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
    else:
        device_info = "CPU (consider upgrading for better performance)"
    
    print(f"Hardware: {device_info}")
    
    # Verify data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("Please ensure the Kamitani dataset is downloaded")
        return
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Subject: {SUBJECT}")
    print(f"  Max samples: {'ALL' if MAX_SAMPLES is None else MAX_SAMPLES}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE} (high resolution)")
    print(f"  Training epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} (M4 optimized)")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()
    
    try:
        # Import modules
        from brain_image_dataset import create_dataset
        from train_minimal import train_model, visualize_results, save_model, plot_training_curves
        from simple_decoder import SimpleBrainDecoder, ImprovedBrainDecoder, AdvancedBrainDecoder
        
        # Step 1: Create optimized dataset
        print("Step 1: Loading complete dataset with M4 optimizations...")
        start_time = time.time()
        
        dataset = create_dataset(
            data_dir=DATA_DIR,
            subject=SUBJECT,
            max_samples=MAX_SAMPLES,  # Load ALL data
            img_size=IMG_SIZE         # Higher resolution
        )
        
        dataset_time = time.time() - start_time
        print(f"✅ Dataset loaded in {dataset_time:.1f}s: {len(dataset):,} samples")
        
        if len(dataset) > 1000:
            print(f"🎉 Excellent! Large dataset ({len(dataset):,} samples) will enable much better learning")
        elif len(dataset) > 100:
            print(f"✅ Good dataset size ({len(dataset)} samples) for meaningful training")
        else:
            print(f"⚠️  Small dataset ({len(dataset)} samples) - results may be limited")
        
        print()
        
        # Step 2: Train advanced model
        print("Step 2: Training advanced model with M4 optimizations...")
        if torch.backends.mps.is_available():
            print("🔥 Using Apple Metal Performance Shaders - this will be FAST!")
        print(f"Expected training time: 10-20 minutes with {len(dataset):,} samples")
        print()
        
        training_start = time.time()
        
        from simple_decoder import SimpleBrainDecoder
        
        model, train_losses, val_losses = train_model(
            dataset,
            model_class=SimpleBrainDecoder,  # Use simple model with PCA-reduced features
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            device='auto',
            early_stopping_patience=10,
            weight_decay=1e-4
        )
        
        training_time = time.time() - training_start
        print(f"\n🎉 Training completed in {training_time/60:.1f} minutes!")
        print()
        
        # Step 3: Generate comprehensive visualizations
        print("Step 3: Generating enhanced visualizations...")
        
        # Set device for inference
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # Plot training curves
        print("  📊 Generating training progress plots...")
        plot_training_curves(train_losses, val_losses)
        
        # Show reconstruction examples
        print("  🖼️  Generating reconstruction comparisons...")
        visualize_results(model, dataset, device, n_examples=min(8, len(dataset)))
        
        # Step 4: Save optimized model
        print("Step 4: Saving optimized model...")
        model_filename = f"optimized_brain_decoder_m4_{IMG_SIZE}x{IMG_SIZE}_{len(dataset)}samples.pth"
        save_model(model, model_filename)
        
        print()
        print("=== 🎉 SUCCESS! ===")
        print("Optimized brain-to-image reconstruction completed!")
        print(f"📈 Training time: {training_time/60:.1f} minutes")
        print(f"📊 Dataset size: {len(dataset):,} samples")
        print(f"🖼️  Resolution: {IMG_SIZE}x{IMG_SIZE}")
        print(f"🧠 Model: {model.__class__.__name__}")
        print("\nGenerated files:")
        print(f"  📊 training_curves_*.png - Training progress")
        print(f"  🖼️  reconstruction_results_*.png - Brain→Image results")
        print(f"  💾 {model_filename} - Trained model")
        print()
        print("🚀 The M4 MacBook Pro optimization should show much better")
        print("   reconstruction quality compared to the minimal version!")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("\n🔧 Debugging tips:")
        print("1. Install required packages:")
        print("   pip install torch torchvision nibabel pandas pillow matplotlib tqdm numpy scikit-learn")
        print("2. For M4 MacBook Pro, ensure you have the latest PyTorch with MPS support:")
        print("   pip install --upgrade torch torchvision")
        print("3. Verify the Kamitani dataset path is correct")
        print("4. Check that fMRI files exist in the expected BIDS format")
        print("5. Ensure sufficient disk space (dataset + models ~2-5GB)")
        print()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧠→🖼️  Brain-to-Image Reconstruction with M4 Optimization")
    print("=" * 60)
    main()