import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA

class BrainImageDataset(Dataset):
    """Enhanced dataset with multiple hemodynamic delays and better preprocessing"""
    
    def __init__(self, events, fmri_data_list, images, roi_mask=None, 
                 tr=3.0, hemodynamic_delays=[4.0, 6.0, 8.0]):
        """
        Args:
            events: DataFrame with stimulus timing info
            fmri_data_list: List of dicts with fMRI data and metadata
            images: Dict mapping stim_id to image arrays
            roi_mask: 3D boolean mask for ROI (optional)
            tr: TR time in seconds (time between fMRI volumes)
            hemodynamic_delays: List of delays to try for hemodynamic response
        """
        self.events = events.reset_index(drop=True)
        self.fmri_data_list = fmri_data_list
        self.images = images
        self.roi_mask = roi_mask
        self.tr = tr
        self.hemodynamic_delays = hemodynamic_delays
        
        # Create mapping from (session, run) to fMRI data
        self.fmri_lookup = {}
        for fmri_item in fmri_data_list:
            key = (fmri_item['session'], fmri_item['run'])
            self.fmri_lookup[key] = fmri_item['data']
        
        # Filter events to only include stimuli we have images for
        valid_events = []
        for idx, row in self.events.iterrows():
            if row['stim_id'] in images:
                key = (row['session'], row['run'])
                if key in self.fmri_lookup:
                    valid_events.append(idx)
        
        self.valid_indices = valid_events
        print(f"BrainImageDataset created with {len(self.valid_indices)} valid samples")
        
        # Pre-extract all fMRI features for faster training
        print("Pre-extracting fMRI features...")
        raw_features = self._preextract_features()
        
        # Apply PCA to reduce dimensionality dramatically
        print("Applying PCA dimensionality reduction for M4 speed...")
        all_features = np.array([raw_features[idx] for idx in self.valid_indices])
        print(f"Original features: {all_features.shape[1]:,} voxels")
        
        self.pca = PCA(n_components=min(100, len(self.valid_indices)-1))  # Max 100 components
        reduced_features = self.pca.fit_transform(all_features)
        
        print(f"PCA reduced to: {reduced_features.shape[1]} features")
        print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Cache the reduced features
        self.fmri_features_cache = {}
        for i, idx in enumerate(self.valid_indices):
            self.fmri_features_cache[idx] = reduced_features[i]
        
    def _preextract_features(self):
        """Pre-extract and cache fMRI features for all samples"""
        features_cache = {}
        
        for idx in self.valid_indices:
            event = self.events.iloc[idx]
            key = (event['session'], event['run'])
            fmri_data = self.fmri_lookup[key]
            
            # Try multiple hemodynamic delays and average
            delay_features = []
            for delay in self.hemodynamic_delays:
                fmri_time = event['onset'] + delay
                time_idx = min(int(fmri_time / self.tr), fmri_data.shape[3] - 1)
                time_idx = max(0, time_idx)  # Ensure non-negative
                
                fmri_vol = fmri_data[:, :, :, time_idx]
                
                if self.roi_mask is not None:
                    features = fmri_vol[self.roi_mask].flatten()
                else:
                    features = fmri_vol.flatten()
                
                delay_features.append(features)
            
            # Average across hemodynamic delays
            avg_features = np.mean(delay_features, axis=0)
            features_cache[idx] = avg_features
        
        return features_cache
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        event_idx = self.valid_indices[idx]
        event = self.events.iloc[event_idx]
        
        # Get pre-extracted features
        fmri_features = self.fmri_features_cache[event_idx]
        
        # Get corresponding image
        stim_id = event['stim_id']
        image = self.images[stim_id] / 255.0  # Normalize to [0,1]
        
        # Convert to tensors
        fmri_tensor = torch.FloatTensor(fmri_features)
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)  # (C, H, W)
        
        return fmri_tensor, image_tensor, str(stim_id)

def create_dataset(data_dir, subject='sub-01', max_samples=None, img_size=128):
    """Create optimized dataset with full data loading"""
    from minimal_data_loader import MinimalKamitaniLoader
    from synthetic_images import create_synthetic_images
    
    # Load ALL fMRI data
    loader = MinimalKamitaniLoader(data_dir, subject, max_samples)
    events, fmri_data_list = loader.load_training_data()
    roi_mask = loader.extract_roi_mask()
    
    # Create synthetic images for all unique stimulus IDs
    stim_ids = events['stim_id'].dropna().unique()
    print(f"\nCreating {len(stim_ids)} synthetic images at {img_size}x{img_size}")
    images = create_synthetic_images(stim_ids, img_size=img_size)
    
    # Create optimized dataset
    dataset = BrainImageDataset(events, fmri_data_list, images, roi_mask)
    
    return dataset