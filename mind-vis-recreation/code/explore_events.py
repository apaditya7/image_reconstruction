# explore_events.py
import pandas as pd
import os

data_dir = "ds001246-download"
subject = "sub-01"

# Look at training session events
train_events_dir = f"{data_dir}/{subject}/ses-perceptionTraining/func"
if os.path.exists(train_events_dir):
    event_files = [f for f in os.listdir(train_events_dir) if f.endswith('_events.tsv')]
    print(f"Training event files: {len(event_files)}")
    
    if event_files:
        # Load first event file
        events = pd.read_csv(f"{train_events_dir}/{event_files[0]}", sep='\t')
        print("\nTraining events structure:")
        print(events.head(10))
        print(f"\nUnique stimulus IDs in this run: {events['stim_id'].nunique()}")
        print(f"Example stimulus IDs: {events['stim_id'].dropna().unique()[:5]}")

# Look at test session events  
test_events_dir = f"{data_dir}/{subject}/ses-perceptionTest01/func"
if os.path.exists(test_events_dir):
    event_files = [f for f in os.listdir(test_events_dir) if f.endswith('_events.tsv')]
    print(f"\nTest event files: {len(event_files)}")
    
    if event_files:
        events = pd.read_csv(f"{test_events_dir}/{event_files[0]}", sep='\t')
        print("\nTest events structure:")
        print(events.head(10))
        print(f"\nUnique stimulus IDs in this run: {events['stim_id'].nunique()}")