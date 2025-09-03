# explore_rois.py
import os

sourcedata_dir = "ds001246-download/sourcedata"

if os.path.exists(sourcedata_dir):
    print("ROI masks available:")
    for item in os.listdir(sourcedata_dir):
        if item.startswith('sub-'):
            roi_dir = f"{sourcedata_dir}/{item}"
            if os.path.exists(roi_dir):
                roi_files = os.listdir(roi_dir)
                print(f"\n{item}: {len(roi_files)} ROI files")
                for roi_file in roi_files[:5]:  # First 5
                    print(f"  {roi_file}")