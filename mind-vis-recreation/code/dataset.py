import numpy as np
import torch
from torch.utils.data import Dataset

class SimpleKamitaniDataset(Dataset):
    def __init__(self, data_path, subject='sbj_3'):
        # Load one subject's data
        data = np.load(f"{data_path}/{subject}.npz")
        
        # Get visual cortex voxels only
        vc_mask = data['VC'] 
        self.train_fmri = data['arr_0'][:, vc_mask]  # Training fMRI
        self.test_fmri = data['arr_2'][:, vc_mask]   # Test fMRI
        
        # Load corresponding images
        images = np.load(f"{data_path}/images_256.npz")
        self.train_images = images['train_images'][data['arr_3']]
        self.test_images = images['test_images']
        
        # Simple normalization
        self.train_fmri = (self.train_fmri - self.train_fmri.mean()) / self.train_fmri.std()
        
    def __len__(self):
        return len(self.train_fmri)
    
    def __getitem__(self, idx):
        return {
            'fmri': torch.FloatTensor(self.train_fmri[idx]),
            'image': torch.FloatTensor(self.train_images[idx])
        }