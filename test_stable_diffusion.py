#!/usr/bin/env python3
"""
Test Stable Diffusion loading separately to diagnose the issue
"""
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import traceback

def test_stable_diffusion():
    print("=== Testing Stable Diffusion Loading ===")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Cache directory
    cache_dir = Path.home() / ".cache" / "brain_decoder" / "stable_diffusion"
    print(f"Cache dir: {cache_dir}")
    
    try:
        print("1. Loading Stable Diffusion pipeline...")
        
        if device == "cuda":
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                cache_dir=cache_dir
            )
        
        print("2. Moving to device...")
        pipe = pipe.to(device)
        
        print("3. Testing basic generation...")
        # Test with a simple text prompt first
        test_image = pipe(
            "a red circle",
            num_inference_steps=10,  # Fewer steps for testing
            guidance_scale=7.5
        ).images[0]
        
        print("4. Saving test image...")
        test_image.save("test_stable_diffusion.png")
        
        print("✅ SUCCESS: Stable Diffusion is working!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_stable_diffusion()