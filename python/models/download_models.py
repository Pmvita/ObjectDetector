#!/usr/bin/env python3
"""
Download COCO-SSD model files for object detection.
Downloads MobileNet-SSD Caffe model files for use with OpenCV DNN.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Model URLs for MobileNet-SSD (Caffe format)
# Try multiple sources in case one is down
MODEL_URLS = [
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.caffemodel",
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
]

CONFIG_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt",
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.prototxt",
]

# Model directory (same as detector.py expects)
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "MobileNetSSD_deploy.caffemodel"
CONFIG_PATH = MODEL_DIR / "MobileNetSSD_deploy.prototxt"

def download_file(urls: list, dest_path: Path, description: str = "") -> bool:
    """Download a file from URL to destination path, trying multiple URLs."""
    if dest_path.exists():
        print(f"✓ {description} already exists: {dest_path}")
        return True
    
    print(f"Downloading {description}...")
    
    for i, url in enumerate(urls, 1):
        try:
            print(f"  Trying source {i}/{len(urls)}: {url}")
            print(f"  Destination: {dest_path}")
            
            # Show progress
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
                print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, dest_path, show_progress)
            print(f"\n✓ Downloaded {description} successfully")
            return True
        except Exception as e:
            print(f"\n  ✗ Failed: {e}")
            if i < len(urls):
                print(f"  Trying next source...")
            continue
    
    return False

def create_prototxt_file(dest_path: Path) -> bool:
    """Create the prototxt file from our local template."""
    try:
        # Try to copy from our local template file
        template_path = Path(__file__).parent / "mobilenet_ssd.prototxt"
        if template_path.exists():
            import shutil
            shutil.copy(template_path, dest_path)
            print(f"✓ Created prototxt file from template")
            return True
        
        # If template doesn't exist, try OpenCV samples
        try:
            import cv2
            import os
            
            opencv_path = os.path.dirname(cv2.__file__)
            samples_path = os.path.join(opencv_path, '..', '..', 'share', 'opencv4', 'samples', 'dnn')
            
            prototxt_sample = None
            for root, dirs, files in os.walk(samples_path):
                if 'MobileNetSSD_deploy.prototxt' in files:
                    prototxt_sample = os.path.join(root, 'MobileNetSSD_deploy.prototxt')
                    break
            
            if prototxt_sample and os.path.exists(prototxt_sample):
                import shutil
                shutil.copy(prototxt_sample, dest_path)
                print(f"✓ Copied prototxt from OpenCV samples")
                return True
        except:
            pass
        
        print("\n⚠ Could not create prototxt file automatically.")
        return False
    except Exception as e:
        print(f"\n✗ Error creating prototxt: {e}")
        return False

def main():
    """Main function to download model files."""
    print("=" * 60)
    print("COCO-SSD Model Downloader")
    print("=" * 60)
    print()
    
    # Check if files already exist
    model_exists = MODEL_PATH.exists()
    config_exists = CONFIG_PATH.exists()
    
    if model_exists and config_exists:
        print("✓ Model files already exist!")
        print(f"  Model: {MODEL_PATH}")
        print(f"  Config: {CONFIG_PATH}")
        print("\nYou can now run detector.py to start object detection.")
        return
    
    print("Downloading MobileNet-SSD model files...")
    print("(This may take a few minutes depending on your connection)")
    print()
    
    # Download config file first (smaller)
    print("Step 1: Downloading config file (prototxt)...")
    config_success = download_file(
        CONFIG_URLS,
        CONFIG_PATH,
        "Config file (prototxt)"
    )
    
    if not config_success:
        print("\n⚠ Could not download config file from any source.")
        print("Attempting to create prototxt file from local template...")
        if create_prototxt_file(CONFIG_PATH):
            print("✓ Created prototxt file successfully!")
        else:
            print("\n✗ Failed to create config file.")
            print("\nThe prototxt file is required. Please ensure the template exists.")
            sys.exit(1)
    
    print()
    
    # Download model file (larger, ~23MB)
    print("Step 2: Downloading model file (caffemodel, ~23MB)...")
    model_success = download_file(
        MODEL_URLS,
        MODEL_PATH,
        "Model file (caffemodel)"
    )
    
    if not model_success:
        print("\n✗ Failed to download model file from all sources.")
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*60)
        print("\nThe model weights file (~23MB) needs to be downloaded manually.")
        print("\nOption 1 - Using wget/curl:")
        print("  Run this command in your terminal:")
        print(f"  cd {MODEL_DIR}")
        print("  wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel")
        print("  mv MobileNetSSD_deploy.caffemodel", MODEL_PATH.name)
        print("\nOption 2 - Manual download:")
        print("  1. Visit: https://github.com/chuanqi305/MobileNet-SSD")
        print("  2. Click on 'MobileNetSSD_deploy.caffemodel'")
        print("  3. Click 'Download' or 'View Raw'")
        print(f"  4. Save to: {MODEL_PATH}")
        print("\nOption 3 - Alternative source:")
        print("  Try searching for 'MobileNet-SSD caffemodel download'")
        print("  or use OpenCV's model zoo")
        print("\n" + "="*60)
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("✓ Model download complete!")
    print("=" * 60)
    print(f"\nModel files location:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Config: {CONFIG_PATH}")
    print("\nYou can now run detector.py to start object detection.")
    print("\nNote: The model file is ~23MB. Make sure you have enough disk space.")

if __name__ == "__main__":
    main()

