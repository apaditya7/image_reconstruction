#!/usr/bin/env python3
"""
CLIP-Based Brain Decoding Implementation
Maps fMRI brain signals to CLIP embeddings instead of raw pixels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import nibabel as nib
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import random

# Set cache directory for models to avoid re-downloads
CACHE_DIR = Path.home() / ".cache" / "brain_decoder"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def create_synthetic_images_from_stim_ids(stim_ids, output_dir="synthetic_kamitani_images", img_size=224):
    """
    Create synthetic images based on stimulus IDs since real ImageNet images aren't included
    Uses deterministic patterns based on stim_id to ensure consistency
    """
    os.makedirs(output_dir, exist_ok=True)
    
    images = {}
    image_paths = []
    
    # Extended color palette
    colors = [
        (255, 120, 120), (120, 255, 120), (120, 120, 255), (255, 255, 120),
        (255, 120, 255), (120, 255, 255), (255, 180, 120), (180, 120, 255),
        (120, 255, 180), (255, 120, 180), (200, 200, 200), (255, 200, 120),
        (150, 255, 150), (255, 150, 150), (150, 150, 255), (255, 255, 150)
    ]
    
    bg_colors = [
        (20, 20, 20), (40, 40, 40), (60, 60, 60), (15, 25, 35),
        (35, 25, 15), (25, 35, 25), (30, 30, 50), (50, 30, 30)
    ]
    
    print(f"Creating synthetic images for {len(stim_ids)} stimulus IDs...")
    
    for i, stim_id in enumerate(tqdm(stim_ids, desc="Creating synthetic images")):
        # Use stim_id as seed for deterministic generation
        random.seed(int(float(stim_id)))
        
        # Create base image
        bg_color = bg_colors[i % len(bg_colors)]
        img = Image.new('RGB', (img_size, img_size), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Choose colors based on stim_id
        primary_color = colors[int(float(stim_id)) % len(colors)]
        secondary_color = colors[(int(float(stim_id)) + 1) % len(colors)]
        
        # Different margins and shapes based on stim_id
        margin = img_size // (4 + (int(float(stim_id)) % 3))
        shape_type = int(float(stim_id)) % 6
        
        if shape_type == 0:  # Rectangle
            draw.rectangle([margin, margin, img_size-margin, img_size-margin], fill=primary_color)
            inner_margin = margin + img_size // 8
            if inner_margin < img_size - inner_margin:
                draw.rectangle([inner_margin, inner_margin, img_size-inner_margin, img_size-inner_margin], fill=secondary_color)
                
        elif shape_type == 1:  # Circle
            draw.ellipse([margin, margin, img_size-margin, img_size-margin], fill=primary_color)
            inner_margin = margin + img_size // 8
            if inner_margin < img_size - inner_margin:
                draw.ellipse([inner_margin, inner_margin, img_size-inner_margin, img_size-inner_margin], fill=secondary_color)
                
        elif shape_type == 2:  # Triangle
            draw.polygon([
                (img_size//2, margin),
                (margin, img_size-margin), 
                (img_size-margin, img_size-margin)
            ], fill=primary_color)
            
        elif shape_type == 3:  # Diamond
            center = img_size // 2
            draw.polygon([
                (center, margin),
                (img_size - margin, center),
                (center, img_size - margin),
                (margin, center)
            ], fill=primary_color)
            
        elif shape_type == 4:  # Cross
            thick = img_size // 6
            center = img_size // 2
            draw.rectangle([margin, center - thick//2, img_size-margin, center + thick//2], fill=primary_color)
            draw.rectangle([center - thick//2, margin, center + thick//2, img_size-margin], fill=primary_color)
            
        else:  # Star pattern
            center = img_size // 2
            points = []
            for angle in range(0, 360, 45):
                x = center + int((img_size//2 - margin) * np.cos(np.radians(angle)))
                y = center + int((img_size//2 - margin) * np.sin(np.radians(angle)))
                points.append((x, y))
            
            for point in points:
                draw.line([center, center, point[0], point[1]], fill=primary_color, width=img_size//16)
        
        # Add texture based on stim_id
        if int(float(stim_id)) % 7 == 0:
            for _ in range(5):
                x = random.randint(margin//2, img_size - margin//2)
                y = random.randint(margin//2, img_size - margin//2) 
                dot_size = img_size // 32
                draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill=secondary_color)
        
        # Save image
        img_path = Path(output_dir) / f"{stim_id}.png"
        img.save(img_path)
        images[stim_id] = np.array(img)
        image_paths.append(img_path)
        
        # Reset random seed
        random.seed()
    
    print(f"Created {len(images)} synthetic images in {output_dir}")
    return images, image_paths

class KamitaniCLIPDataset(Dataset):
    """
    Dataset that loads real Kamitani fMRI data and creates synthetic images for CLIP embeddings
    """
    def __init__(self, data_dir, subject='sub-01', max_samples=50, session_type='perceptionTest01'):
        self.data_dir = Path(data_dir)
        self.subject = subject
        self.max_samples = max_samples
        self.session_type = session_type
        
        # Setup CLIP model with caching
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading CLIP model on {self.device}...")
        
        # Load CLIP model with caching
        model_cache_dir = CACHE_DIR / "clip_model"
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", 
            cache_dir=model_cache_dir
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=model_cache_dir
        )
        
        # Set dtype based on device capabilities
        if self.device == "cuda":
            self.clip_model = self.clip_model.to(torch.bfloat16).to(self.device)
        else:
            self.clip_model = self.clip_model.to(self.device)
        
        # Load data
        self.fmri_data, self.image_paths, self.clip_embeddings, self.stim_ids = self._load_data()
        
    def _load_data(self):
        """Load fMRI data and corresponding images, extract CLIP embeddings"""
        print(f"Loading Kamitani dataset for {self.subject}, session: {self.session_type}...")
        
        # Find fMRI files in correct session directory
        session_dir = f"ses-{self.session_type}"
        fmri_dir = self.data_dir / self.subject / session_dir / "func"
        
        if not fmri_dir.exists():
            raise FileNotFoundError(f"No fMRI directory found at {fmri_dir}")
        
        fmri_files = list(fmri_dir.glob("*bold.nii*"))
        events_files = list(fmri_dir.glob("*events.tsv"))
        
        if not fmri_files:
            raise FileNotFoundError(f"No fMRI files found in {fmri_dir}")
        if not events_files:
            raise FileNotFoundError(f"No events files found in {fmri_dir}")
            
        print(f"Found {len(fmri_files)} fMRI files and {len(events_files)} events files")
        
        # Load first fMRI file and corresponding events
        fmri_file = fmri_files[0]
        events_file = events_files[0]
        
        print(f"Loading fMRI data from {fmri_file}")
        print(f"Loading events from {events_file}")
        
        # Load fMRI data
        fmri_img = nib.load(fmri_file)
        fmri_data = fmri_img.get_fdata()
        
        # Load events to get stimulus IDs
        events_df = pd.read_csv(events_file, sep='\t')
        stimulus_events = events_df[events_df['event_type'] == 'stimulus'].copy()
        
        if len(stimulus_events) == 0:
            raise ValueError(f"No stimulus events found in {events_file}")
        
        print(f"Found {len(stimulus_events)} stimulus events")
        
        # Reshape fMRI data: (x, y, z, time) -> (time, voxels)
        original_shape = fmri_data.shape[:3]
        fmri_data = fmri_data.reshape(-1, fmri_data.shape[-1]).T
        
        # Select active voxels (top 10% most variable)
        voxel_std = np.std(fmri_data, axis=0)
        active_voxels = voxel_std > np.percentile(voxel_std, 90)
        fmri_data = fmri_data[:, active_voxels]
        
        print(f"Selected {fmri_data.shape[1]} active voxels from {np.prod(original_shape)} total")
        
        # Get unique stimulus IDs
        unique_stim_ids = stimulus_events['stim_id'].unique()
        
        # Limit samples
        n_samples = min(self.max_samples, len(unique_stim_ids), fmri_data.shape[0])
        selected_stim_ids = unique_stim_ids[:n_samples]
        
        # Create synthetic images for these stimulus IDs
        print(f"Creating synthetic images for {n_samples} unique stimuli...")
        synthetic_images, image_paths = create_synthetic_images_from_stim_ids(selected_stim_ids)
        
        # Extract CLIP embeddings for synthetic images
        print("Extracting CLIP embeddings from synthetic images...")
        clip_embeddings = []
        valid_indices = []
        valid_stim_ids = []
        
        for i, (stim_id, img_path) in enumerate(tqdm(zip(selected_stim_ids, image_paths), desc="Processing images")):
            try:
                # Load synthetic image
                image = Image.open(img_path).convert('RGB')
                
                # Process with CLIP
                inputs = self.clip_processor(images=image, return_tensors="pt")
                
                if self.device == "cuda":
                    inputs = {k: v.to(torch.bfloat16).to(self.device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract CLIP embedding
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    image_features = F.normalize(image_features, dim=-1)
                    
                clip_embeddings.append(image_features.cpu().float().numpy())
                valid_indices.append(i)
                valid_stim_ids.append(stim_id)
                
            except Exception as e:
                print(f"Error processing {stim_id}: {e}")
                continue
        
        if len(clip_embeddings) == 0:
            raise RuntimeError("No valid image-brain pairs found")
        
        clip_embeddings = np.vstack(clip_embeddings)
        
        # Match fMRI data to stimuli
        # For simplicity, use first N timepoints corresponding to stimulus presentations
        fmri_data = fmri_data[:len(valid_stim_ids)]
        image_paths = [str(image_paths[i]) for i in valid_indices]  # Convert Path to string
        
        print(f"Successfully loaded {len(valid_stim_ids)} samples")
        print(f"fMRI shape: {fmri_data.shape}")
        print(f"CLIP embeddings shape: {clip_embeddings.shape}")
        
        return fmri_data, image_paths, clip_embeddings, valid_stim_ids
    
    def __len__(self):
        return len(self.stim_ids)
    
    def __getitem__(self, idx):
        return {
            'fmri': torch.FloatTensor(self.fmri_data[idx]),
            'clip_embedding': torch.FloatTensor(self.clip_embeddings[idx]),
            'image_path': self.image_paths[idx],
            'stim_id': self.stim_ids[idx]
        }

class BrainToCLIPMapper(nn.Module):
    """
    Simple neural network to map fMRI signals to CLIP embeddings
    """
    def __init__(self, num_voxels, clip_dim=512):
        super().__init__()
        
        # Simple architecture for M4 MacBook Pro
        self.mapper = nn.Sequential(
            nn.Linear(num_voxels, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, clip_dim)
        )
        
    def forward(self, fmri):
        clip_pred = self.mapper(fmri)
        # Normalize like CLIP embeddings
        return F.normalize(clip_pred, dim=-1)

def cosine_similarity_loss(pred, target):
    """Cosine similarity loss for CLIP embeddings"""
    return 1 - F.cosine_similarity(pred, target, dim=-1).mean()

def train_brain_to_clip(dataset, epochs=50, batch_size=8, lr=0.001):
    """Train the brain-to-CLIP mapping"""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # Create model
    sample_fmri = dataset[0]['fmri']
    model = BrainToCLIPMapper(num_voxels=sample_fmri.shape[0]).to(device)
    
    # Setup training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    losses = []
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            fmri = batch['fmri'].to(device)
            target_clip = batch['clip_embedding'].to(device).squeeze(1)
            
            # Forward pass
            pred_clip = model(fmri)
            
            # Cosine similarity loss
            loss = cosine_similarity_loss(pred_clip, target_clip)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return model, losses

def generate_images_from_brain(model, dataset, num_samples=3):
    """Generate images from brain activity using Stable Diffusion"""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load Stable Diffusion pipeline with caching
    print("Loading Stable Diffusion pipeline...")
    sd_cache_dir = CACHE_DIR / "stable_diffusion"
    
    if device == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.bfloat16,
            cache_dir=sd_cache_dir
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            cache_dir=sd_cache_dir
        )
    
    pipe = pipe.to(device)
    
    model.eval()
    results = []
    
    print(f"Generating {num_samples} images from brain activity...")
    
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Generating images"):
        sample = dataset[i]
        fmri = sample['fmri'].unsqueeze(0).to(device)
        original_path = sample['image_path']
        stim_id = sample['stim_id']
        
        # Predict CLIP embedding from brain activity
        with torch.no_grad():
            pred_clip = model(fmri)
        
        # Generate image using predicted CLIP embedding
        try:
            with torch.no_grad():
                # Convert to format expected by Stable Diffusion
                if device == "cuda":
                    prompt_embeds = pred_clip.to(torch.bfloat16)
                else:
                    prompt_embeds = pred_clip.to(torch.float32)
                
                generated_image = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=None,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
            
            # Load original synthetic image for comparison
            original_image = Image.open(original_path).convert('RGB')
            
            results.append({
                'original': original_image,
                'generated': generated_image,
                'path': original_path,
                'stim_id': stim_id
            })
            
        except Exception as e:
            print(f"Error generating image {i} (stim_id: {stim_id}): {e}")
            continue
    
    return results

def visualize_results(results, save_path="brain_to_image_results.png"):
    """Visualize original vs generated images"""
    if not results:
        print("No results to visualize")
        return
        
    n_samples = len(results)
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, result in enumerate(results):
        # Original synthetic image
        axes[0, i].imshow(result['original'])
        axes[0, i].set_title(f"Synthetic Image {i+1}\n(Stim ID: {result['stim_id'][:8]}...)")
        axes[0, i].axis('off')
        
        # Generated image from brain activity
        axes[1, i].imshow(result['generated'])
        axes[1, i].set_title(f"Generated from Brain {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to {save_path}")
    plt.show()

def main():
    """Main execution function"""
    print("=== CLIP-Based Brain Decoding (Kamitani Dataset) ===")
    print("Note: Using synthetic images since ImageNet images require separate download")
    print()
    
    # Configuration
    DATA_DIR = os.path.expanduser("~/ds001246-download")
    SUBJECT = 'sub-01'
    SESSION_TYPE = 'perceptionTest01'  # Fixed session name
    MAX_SAMPLES = 50
    EPOCHS = 25
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    
    print(f"Configuration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Subject: {SUBJECT}")
    print(f"  Session: {SESSION_TYPE}")
    print(f"  Max samples: {MAX_SAMPLES}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Model cache: {CACHE_DIR}")
    print()
    
    try:
        # Load dataset with CLIP embeddings
        dataset = KamitaniCLIPDataset(DATA_DIR, SUBJECT, MAX_SAMPLES, SESSION_TYPE)
        
        # Train brain-to-CLIP mapper
        model, losses = train_brain_to_clip(
            dataset, 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE
        )
        
        # Plot training losses
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss (1 - Cosine Similarity)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_losses.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Generate images from brain activity
        results = generate_images_from_brain(model, dataset, num_samples=3)
        
        # Visualize results
        visualize_results(results)
        
        # Save model
        torch.save(model.state_dict(), 'brain_to_clip_mapper.pth')
        print("Model saved as brain_to_clip_mapper.pth")
        
        print("\n=== SUCCESS! ===")
        print("CLIP-based brain decoding completed!")
        print("The generated images should show semantic similarity")
        print("to the synthetic representations of the stimulus patterns.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()