import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientBrainDecoder(nn.Module):
    """Lightweight decoder optimized for M4 MacBook Pro - much faster training"""
    
    def __init__(self, n_voxels, img_size=128):
        super().__init__()
        self.img_size = img_size
        
        # Lightweight fMRI encoder - much smaller than before
        self.fmri_encoder = nn.Sequential(
            nn.Linear(n_voxels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256)  # Much smaller bottleneck
        )
        
        # Efficient decoder for 128x128
        if img_size == 128:
            # 8x8 -> 128x128 in 4 steps (8->16->32->64->128)
            self.decoder = nn.Sequential(
                nn.Linear(256, 128 * 8 * 8),  # Start with 128 channels, not 512
                nn.ReLU(),
                nn.Unflatten(1, (128, 8, 8)),
                
                # 8x8 -> 16x16
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                # 16x16 -> 32x32
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                # 32x32 -> 64x64
                nn.ConvTranspose2d(32, 16, 4, 2, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                
                # 64x64 -> 128x128
                nn.ConvTranspose2d(16, 8, 4, 2, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                
                # Final layer to RGB
                nn.Conv2d(8, 3, 3, 1, 1),
                nn.Sigmoid()
            )
        else:  # 64x64
            self.decoder = nn.Sequential(
                nn.Linear(256, 64 * 8 * 8),
                nn.ReLU(),
                nn.Unflatten(1, (64, 8, 8)),
                
                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16x16
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 32x32
                nn.BatchNorm2d(16),
                nn.ReLU(),
                
                nn.ConvTranspose2d(16, 8, 4, 2, 1),   # 64x64
                nn.BatchNorm2d(8),
                nn.ReLU(),
                
                nn.Conv2d(8, 3, 3, 1, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Efficient weight initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, fmri_features):
        # Encode fMRI features
        encoded = self.fmri_encoder(fmri_features)
        
        # Decode to image
        images = self.decoder(encoded)
        
        return images

class FastBrainDecoder(nn.Module):
    """Super fast decoder for M4 MacBook Pro - designed for speed"""
    
    def __init__(self, n_voxels, img_size=64):  # Start with 64x64 for speed
        super().__init__()
        self.img_size = img_size
        
        # Aggressive dimensionality reduction first
        self.voxel_reducer = nn.Sequential(
            nn.Linear(n_voxels, 2048),  # Huge reduction from ~100k to 2k
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # Very compact representation
        )
        
        # Small CNN decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 32 * 8 * 8),  # Very small start
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            
            # 16x16 -> 32x32  
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(8, 3, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fmri_features):
        # Reduce dimensionality aggressively
        reduced = self.voxel_reducer(fmri_features)
        
        # Decode to image
        images = self.decoder(reduced)
        
        return images

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model sizes
    n_voxels = 100000  # Typical for whole brain
    
    efficient = EfficientBrainDecoder(n_voxels, 128)
    fast = FastBrainDecoder(n_voxels, 64)
    
    print(f"EfficientBrainDecoder (128x128) parameters: {count_parameters(efficient):,}")
    print(f"FastBrainDecoder (64x64) parameters: {count_parameters(fast):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, n_voxels)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        out_efficient = efficient(dummy_input)
        out_fast = fast(dummy_input)
        
        print(f"EfficientBrainDecoder output: {out_efficient.shape}")
        print(f"FastBrainDecoder output: {out_fast.shape}")
        
        # Speed test
        import time
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        fast_mps = fast.to(device)
        dummy_mps = dummy_input.to(device)
        
        # Warmup
        for _ in range(5):
            _ = fast_mps(dummy_mps)
        
        torch.mps.synchronize() if device.type == 'mps' else None
        
        start = time.time()
        for _ in range(10):
            _ = fast_mps(dummy_mps)
        
        torch.mps.synchronize() if device.type == 'mps' else None
        end = time.time()
        
        print(f"FastBrainDecoder on {device}: {(end-start)/10*1000:.1f}ms per batch")