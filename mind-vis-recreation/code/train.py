import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SimpleKamitaniDataset
from fmri_encoder import SimpleFMRIEncoder
from image_decoder import SimpleImageDecoder

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SimpleKamitaniDataset('data/Kamitani')
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Models
num_voxels = dataset.train_fmri.shape[1]
encoder = SimpleFMRIEncoder(num_voxels).to(device)
decoder = SimpleImageDecoder().to(device)

# Training
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), 
    lr=1e-4
)
criterion = nn.MSELoss()

for epoch in range(100):
    total_loss = 0
    for batch in loader:
        fmri = batch['fmri'].to(device)
        images = batch['image'].to(device)
        
        # Resize images to match decoder output
        images = torch.nn.functional.interpolate(images.permute(0,3,1,2), size=64)
        images = (images - 0.5) * 2  # Normalize to [-1, 1]
        
        # Forward pass
        features = encoder(fmri)
        pred_images = decoder(features)
        
        # Loss
        loss = criterion(pred_images, images)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")