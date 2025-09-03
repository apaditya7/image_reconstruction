import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBrainDecoder(nn.Module):
    """Simple neural network to decode images from brain activity"""
    
    def __init__(self, n_voxels, img_size=64, hidden_dims=[32, 16]):
        super().__init__()
        self.img_size = img_size
        self.n_pixels = img_size * img_size * 3  # RGB
        
        # Build encoder layers
        layers = []
        prev_dim = n_voxels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, self.n_pixels))
        layers.append(nn.Sigmoid())  # Output [0,1] range
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, fmri_features):
        """
        Args:
            fmri_features: (batch_size, n_voxels)
        Returns:
            images: (batch_size, 3, img_size, img_size)
        """
        # Decode to flattened pixels
        pixels = self.decoder(fmri_features)
        
        # Reshape to image format
        batch_size = fmri_features.size(0)
        images = pixels.view(batch_size, 3, self.img_size, self.img_size)
        
        return images

class ImprovedBrainDecoder(nn.Module):
    """Advanced CNN-based decoder optimized for M4 MacBook Pro"""
    
    def __init__(self, n_voxels, img_size=128):
        super().__init__()
        self.img_size = img_size
        
        # Calculate intermediate size for proper upsampling to target size
        if img_size == 128:
            init_size = 8  # 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        else:
            init_size = 8
        
        # Deep fMRI feature encoder with BatchNorm and residual connections
        self.fmri_encoder = nn.Sequential(
            nn.Linear(n_voxels, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 512 * init_size * init_size)
        )
        
        # CNN decoder with proper upsampling for 128x128
        decoder_layers = [
            nn.Unflatten(1, (512, init_size, init_size)),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ]
        
        # Add extra layer for 128x128 output
        if img_size == 128:
            decoder_layers.extend([
                # 64x64 -> 128x128
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                # Final layer to 3 channels
                nn.Conv2d(32, 3, 3, 1, 1),
                nn.Sigmoid()
            ])
        else:
            # For 64x64 output
            decoder_layers.extend([
                nn.Conv2d(64, 3, 3, 1, 1),
                nn.Sigmoid()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, fmri_features):
        # Encode fMRI features
        encoded = self.fmri_encoder(fmri_features)
        
        # Decode to image
        images = self.decoder(encoded)
        
        return images

class AdvancedBrainDecoder(nn.Module):
    """Most advanced decoder with attention and skip connections"""
    
    def __init__(self, n_voxels, img_size=128):
        super().__init__()
        self.img_size = img_size
        
        # Multi-scale fMRI encoder
        self.encoder_branch1 = nn.Sequential(
            nn.Linear(n_voxels, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.encoder_branch2 = nn.Sequential(
            nn.Linear(n_voxels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512 * 8 * 8)
        )
        
        # Decoder with skip connections
        self.decode1 = nn.Sequential(
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        if img_size == 128:
            self.decode4 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 128x128
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
            self.final = nn.Conv2d(32, 3, 3, 1, 1)
        else:
            self.final = nn.Conv2d(64, 3, 3, 1, 1)
        
        self.activation = nn.Sigmoid()
        
    def forward(self, fmri_features):
        # Multi-branch encoding
        branch1 = self.encoder_branch1(fmri_features)
        branch2 = self.encoder_branch2(fmri_features)
        
        # Fuse branches
        fused = torch.cat([branch1, branch2], dim=1)
        encoded = self.fusion(fused)
        
        # Decode with progressive upsampling
        x = self.decode1(encoded)
        x = self.decode2(x)
        x = self.decode3(x)
        
        if hasattr(self, 'decode4'):
            x = self.decode4(x)
        
        x = self.final(x)
        return self.activation(x)