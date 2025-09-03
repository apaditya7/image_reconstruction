import numpy as np
from PIL import Image, ImageDraw
import os
import torch

def create_synthetic_images(stim_ids, output_dir="synthetic_images", img_size=128):
    """
    Create enhanced synthetic images for each stimulus ID at higher resolution
    More complex shapes, patterns and colors for better learning
    """
    os.makedirs(output_dir, exist_ok=True)
    
    images = {}
    
    # Extended color palette with more variety
    colors = [
        (255, 120, 120),  # Light Red
        (120, 255, 120),  # Light Green  
        (120, 120, 255),  # Light Blue
        (255, 255, 120),  # Light Yellow
        (255, 120, 255),  # Light Magenta
        (120, 255, 255),  # Light Cyan
        (255, 180, 120),  # Orange
        (180, 120, 255),  # Purple
        (120, 255, 180),  # Mint
        (255, 120, 180),  # Pink
        (200, 200, 200),  # Gray
        (255, 200, 120),  # Peach
    ]
    
    # Background colors for more variety
    bg_colors = [
        (20, 20, 20),     # Very dark
        (40, 40, 40),     # Dark gray
        (60, 60, 60),     # Medium gray
        (15, 25, 35),     # Dark blue-ish
        (35, 25, 15),     # Dark brown-ish
        (25, 35, 25),     # Dark green-ish
    ]
    
    for i, stim_id in enumerate(stim_ids):
        # Create base image with varied background
        bg_color = bg_colors[i % len(bg_colors)]
        img = Image.new('RGB', (img_size, img_size), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Choose colors
        primary_color = colors[i % len(colors)]
        secondary_color = colors[(i + 1) % len(colors)]
        
        # Different margins for size variety
        margin_sizes = [img_size // 5, img_size // 4, img_size // 3]
        margin = margin_sizes[i % len(margin_sizes)]
        
        shape_type = i % 6  # 6 different shape types now
        
        if shape_type == 0:  # Rectangle
            draw.rectangle([margin, margin, img_size-margin, img_size-margin], fill=primary_color)
            # Add smaller inner rectangle
            inner_margin = margin + img_size // 8
            if inner_margin < img_size - inner_margin:
                draw.rectangle([inner_margin, inner_margin, img_size-inner_margin, img_size-inner_margin], fill=secondary_color)
                
        elif shape_type == 1:  # Circle
            draw.ellipse([margin, margin, img_size-margin, img_size-margin], fill=primary_color)
            # Add smaller inner circle
            inner_margin = margin + img_size // 8
            if inner_margin < img_size - inner_margin:
                draw.ellipse([inner_margin, inner_margin, img_size-inner_margin, img_size-inner_margin], fill=secondary_color)
                
        elif shape_type == 2:  # Triangle
            draw.polygon([
                (img_size//2, margin),
                (margin, img_size-margin), 
                (img_size-margin, img_size-margin)
            ], fill=primary_color)
            
        elif shape_type == 3:  # Diamond
            center = img_size // 2
            size = img_size - 2 * margin
            draw.polygon([
                (center, margin),
                (img_size - margin, center),
                (center, img_size - margin),
                (margin, center)
            ], fill=primary_color)
            
        elif shape_type == 4:  # Cross/Plus
            thick = img_size // 6
            # Horizontal bar
            draw.rectangle([margin, center - thick//2, img_size-margin, center + thick//2], fill=primary_color)
            # Vertical bar
            draw.rectangle([center - thick//2, margin, center + thick//2, img_size-margin], fill=primary_color)
            
        else:  # Star-like pattern (shape_type == 5)
            center = img_size // 2
            points = []
            for angle in range(0, 360, 45):  # 8 points
                x = center + int((img_size//2 - margin) * np.cos(np.radians(angle)))
                y = center + int((img_size//2 - margin) * np.sin(np.radians(angle)))
                points.append((x, y))
            
            # Draw lines from center to each point
            for point in points:
                draw.line([center, center, point[0], point[1]], fill=primary_color, width=img_size//16)
        
        # Add some texture/noise for more complexity
        if i % 7 == 0:  # Occasionally add small dots
            for _ in range(5):
                x = np.random.randint(margin//2, img_size - margin//2)
                y = np.random.randint(margin//2, img_size - margin//2)
                dot_size = img_size // 32
                draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill=secondary_color)
        
        # Save and store
        img_path = f"{output_dir}/{stim_id}.png"
        img.save(img_path)
        images[stim_id] = np.array(img)
        
    print(f"Created {len(images)} enhanced synthetic images ({img_size}x{img_size}) in {output_dir}")
    return images

def normalize_image(img_array):
    """Normalize image to [0,1] range"""
    return img_array.astype(np.float32) / 255.0

def image_to_tensor(img_array):
    """Convert image array to PyTorch tensor (C, H, W)"""
    if len(img_array.shape) == 3:  # (H, W, C)
        return torch.FloatTensor(img_array).permute(2, 0, 1)  # (C, H, W)
    return torch.FloatTensor(img_array)