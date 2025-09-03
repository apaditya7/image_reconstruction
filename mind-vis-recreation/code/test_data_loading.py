# test_data_loading.py
from dataset import SimpleKamitaniDataset
from torch.utils.data import DataLoader

dataset = SimpleKamitaniDataset('data/Kamitani')
loader = DataLoader(dataset, batch_size=4)

for batch in loader:
    print(f"fMRI shape: {batch['fmri'].shape}")
    print(f"Image shape: {batch['image'].shape}")
    break