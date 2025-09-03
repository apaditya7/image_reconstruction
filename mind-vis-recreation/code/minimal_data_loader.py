import nibabel as nib
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MinimalKamitaniLoader:
    """Load complete Kamitani dataset with M4 MacBook Pro optimizations"""
    
    def __init__(self, data_dir, subject='sub-01', max_train_samples=None, 
                 hemodynamic_delays=[4.0, 6.0, 8.0]):
        self.data_dir = data_dir
        self.subject = subject  
        self.max_train_samples = max_train_samples
        self.subject_dir = f"{data_dir}/{subject}"
        self.hemodynamic_delays = hemodynamic_delays
        self.scaler = StandardScaler()
        
    def load_training_data(self):
        """Load ALL training runs from ALL training sessions"""
        # Find all training sessions
        training_sessions = sorted([d for d in os.listdir(self.subject_dir) 
                                  if d.startswith('ses-perceptionTraining')])
        
        if not training_sessions:
            raise FileNotFoundError(f"No perception training sessions found in {self.subject_dir}")
        
        print(f"Found {len(training_sessions)} training sessions: {training_sessions}")
        
        all_events = []
        all_fmri_data = []
        
        for session in training_sessions:
            session_dir = f"{self.subject_dir}/{session}/func"
            print(f"\nProcessing session: {session}")
            
            # Get ALL event files for this session
            event_files = sorted([f for f in os.listdir(session_dir) if f.endswith('_events.tsv')])
            print(f"Found {len(event_files)} runs in {session}")
            
            for event_file in event_files:
                run_name = event_file.replace('_events.tsv', '')
                print(f"Loading run: {run_name}")
                
                # Load events
                events = pd.read_csv(f"{session_dir}/{event_file}", sep='\t')
                
                # Filter to stimulus presentations only
                if 'event_type' in events.columns:
                    stim_events = events[events['event_type'] == 'stimulus'].copy()
                else:
                    stim_events = events.copy()
                
                # Add session and run info for tracking
                stim_events['session'] = session
                stim_events['run'] = run_name
                
                # Load corresponding fMRI data
                fmri_file = event_file.replace('_events.tsv', '_bold.nii.gz')
                fmri_path = f"{session_dir}/{fmri_file}"
                
                try:
                    fmri_img = nib.load(fmri_path)
                    fmri_data = fmri_img.get_fdata()
                    
                    # Store the data with run identifier
                    all_events.append(stim_events)
                    all_fmri_data.append({
                        'data': fmri_data,
                        'session': session,
                        'run': run_name
                    })
                    
                    print(f"  Loaded {len(stim_events)} events, fMRI shape: {fmri_data.shape}")
                    
                except Exception as e:
                    print(f"  Warning: Could not load {fmri_path}: {e}")
                    continue
        
        # Combine all events
        combined_events = pd.concat(all_events, ignore_index=True)
        
        # Limit samples if specified
        if self.max_train_samples is not None:
            combined_events = combined_events.head(self.max_train_samples)
            print(f"\nLimited to {len(combined_events)} samples")
        
        print(f"\nTotal dataset: {len(combined_events)} stimulus events from {len(all_fmri_data)} runs")
        print(f"Unique stimulus IDs: {len(combined_events['stim_id'].unique())}")
        
        return combined_events, all_fmri_data
        
    def extract_roi_mask(self):
        """Load visual cortex ROI mask if available"""
        roi_dir = f"{self.data_dir}/sourcedata/{self.subject}/roi"
        if os.path.exists(roi_dir):
            roi_files = [f for f in os.listdir(roi_dir) if 'VC' in f or 'visual' in f.lower()]
            if roi_files:
                roi_path = f"{roi_dir}/{roi_files[0]}"
                print(f"Loading ROI mask: {roi_path}")
                roi_img = nib.load(roi_path)
                return roi_img.get_fdata() > 0
        
        print("No ROI mask found, using all voxels")
        return None
    
    def normalize_fmri_data(self, fmri_features_list):
        """Apply z-score normalization to fMRI data"""
        print("Applying z-score normalization to fMRI data...")
        
        # Fit scaler on all data
        all_features = np.vstack(fmri_features_list)
        self.scaler.fit(all_features)
        
        # Transform each feature set
        normalized_features = []
        for features in fmri_features_list:
            normalized = self.scaler.transform(features.reshape(1, -1)).flatten()
            normalized_features.append(normalized)
        
        return normalized_features

# Test the loader
if __name__ == "__main__":
    loader = MinimalKamitaniLoader("~/ds001246-download", max_train_samples=None)
    events, fmri_data_list = loader.load_training_data()
    print(f"Loaded {len(events)} events from {len(fmri_data_list)} runs")