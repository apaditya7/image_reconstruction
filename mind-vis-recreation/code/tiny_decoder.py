import torch
import torch.nn as nn

class TinyBrainDecoder(nn.Module):
    """Ultra-tiny decoder for immediate testing - under 1M parameters"""
    
    def __init__(self, n_voxels, img_size=64):
        super().__init__()
        self.img_size = img_size
        
        # Ultra-aggressive dimensionality reduction
        self.encoder = nn.Sequential(
            nn.Linear(n_voxels, 1024),  # From ~100k to 1k
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),       # To 256
            nn.ReLU(), 
            nn.Linear(256, 64)          # To 64 - very compact
        )
        
        # Tiny decoder - direct to pixels
        n_pixels = img_size * img_size * 3
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(), 
            nn.Linear(512, n_pixels),
            nn.Sigmoid()
        )
    
    def forward(self, fmri_features):
        encoded = self.encoder(fmri_features)
        pixels = self.decoder(encoded)
        
        # Reshape to image
        batch_size = fmri_features.size(0)
        images = pixels.view(batch_size, 3, self.img_size, self.img_size)
        
        return images

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    n_voxels = 100000
    model = TinyBrainDecoder(n_voxels, 64)
    
    print(f"TinyBrainDecoder parameters: {count_parameters(model):,}")
    
    # Test
    dummy_input = torch.randn(4, n_voxels)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape}, Output: {output.shape}")
    
    # Speed test on MPS
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        model_mps = model.to(device)
        dummy_mps = dummy_input.to(device)
        
        # Warmup
        for _ in range(5):
            _ = model_mps(dummy_mps)
        
        torch.mps.synchronize()
        
        import time
        start = time.time()
        for _ in range(100):  # More iterations
            _ = model_mps(dummy_mps)
        
        torch.mps.synchronize()
        end = time.time()
        
        print(f"Speed on MPS: {(end-start)/100*1000:.1f}ms per batch of 4")